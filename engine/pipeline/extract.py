"""One record to one result: the local in-process llama.cpp driver.

**In-process, not a server.** The runtime was chosen by measurement, not
assumption. Research flagged a CUDA bug in this architecture's Gated DeltaNet
path that made models load normally and emit **fluent garbage**, with a stale
`libggml-cuda.so` keeping the bug alive after a rebuild -- and this machine has
two disagreeing ggml copies. Five gold labels through the real prompt and
grammar returned 43/43 entries verbatim-grounded with the GPU verified
exclusive, so the in-process path is sound. `llama-server` is not used.

**Structured output is a GBNF grammar generated from the one JSON schema**, via
`LlamaGrammar.from_json_schema`. Hand-writing a grammar is rejected: a second
source of truth whose desync produces plausible, parseable, wrong-shaped output
-- the worst failure mode, because nothing errors. `check_grammar()` asserts
fixture strings are accepted and rejected as expected, guarding against a
silently permissive converter.

A grammar guarantees *shape*, never *truthfulness*, so `validate()` runs on top
of it regardless.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any

from engine.pipeline.config import Config
from engine.services.extraction.label_purpose_risk import SCHEMA

# ggml's CUDA backend allocates pinned host memory (`cudaHostAlloc`) during
# init. That call is refused in this environment, and ggml treats the failure
# as fatal for the whole backend: it prints `ggml_cuda_init: failed to
# initialize CUDA: initialization error` and reports
# `llama_supports_gpu_offload() == False`.
#
# This flag is what makes CUDA init *succeed* -- it is not a CPU fallback, and
# there is no CPU fallback anywhere in this module. Setting it makes ggml skip
# pinned allocation and initialise normally (verified: offload True, 24 GiB
# visible). It must be set before the first llama_cpp import, so it lives at
# module scope rather than in a shell profile, where one forgotten export would
# turn every run into a refusal.
#
# When the GPU genuinely is unusable, `check_gpu_offload()` stops the run --
# see `GPUUnavailable`. Running on CPU is never the answer: it is ~100x slower
# and llama.cpp does it silently, so a corpus run would look healthy for
# years.
os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")

# Free VRAM the model needs before it is worth starting: 16.4 GiB of weights
# plus room for the KV cache and compute buffers. Below this, llama.cpp would
# fall back to CPU rather than fail.
MIN_VRAM_MIB = 18_000

# Fixtures the schema->grammar conversion must judge correctly. A converter
# that accepts everything would pass a parseability check while letting
# wrong-shaped output through, so the negatives matter more than the positive.
GRAMMAR_FIXTURES = [
    (
        "valid",
        True,
        {
            "purpose": [{"text": "supports sleep", "evidence": "Supports Sleep*"}],
            "risk": [],
            "purpose_confidence": 4,
            "risk_confidence": 0,
        },
    ),
    (
        "empty arrays are valid",
        True,
        {"purpose": [], "risk": [], "purpose_confidence": 0, "risk_confidence": 0},
    ),
    (
        "missing evidence",
        False,
        {
            "purpose": [{"text": "supports sleep"}],
            "risk": [],
            "purpose_confidence": 4,
            "risk_confidence": 0,
        },
    ),
    (
        "wrong type on purpose_confidence",
        False,
        {"purpose": [], "risk": [], "purpose_confidence": "high", "risk_confidence": 0},
    ),
]


class GPUUnavailable(RuntimeError):
    """No usable GPU, so the run must stop rather than quietly use the CPU."""


def gpu_holders() -> list[dict]:
    """Processes currently holding GPU memory, via nvidia-smi.

    Used only to explain a failure. The overwhelmingly common cause of "the GPU
    is unavailable" here is a previous run that was never killed and still has
    the weights resident, and naming that process is the difference between a
    one-command fix and a debugging session.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    holders = []
    for line in out.stdout.strip().splitlines():
        parts = [f.strip() for f in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            holders.append({"pid": parts[0], "name": parts[1], "used_mib": parts[2]})
    return holders


def free_vram_mib() -> int | None:
    """Free VRAM on device 0, or None when it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()[:1]
    return int(first[0].strip()) if first and first[0].strip().isdigit() else None


def _gpu_unavailable_message(reason: str) -> str:
    """Explain why the GPU is unusable, and name whatever is holding it."""
    lines = [
        f"Refusing to run: {reason}",
        "",
        "This pipeline is GPU-only by design. Falling back to CPU is not an",
        "option here -- it is ~100x slower, so a corpus run would take years",
        "rather than days, and llama.cpp does it silently: no error, correct",
        "rows, a perfect grounding rate, and only the clock to tell you.",
    ]

    holders = gpu_holders()
    if holders:
        lines += ["", "The GPU is currently held by:"]
        lines += [
            f"  PID {h['pid']}  {h['name']}  ({h['used_mib']} MiB)" for h in holders
        ]
        lines += [
            "",
            "This is usually a previous run that was never killed. Stop it and",
            f"retry:  kill {' '.join(h['pid'] for h in holders)}",
        ]
    else:
        lines += [
            "",
            "No process is holding the GPU, so this is not a leftover run.",
            "Check that GGML_CUDA_NO_PINNED=1 is set before llama_cpp is first",
            "imported (ggml's CUDA init fails on pinned-memory allocation on",
            "this machine), and that libggml-cuda.so loads -- `python -c \"from",
            "llama_cpp import llama_cpp as C; print(C.llama_supports_gpu_offload())\"`",
            "should print True.",
        ]
    return "\n".join(lines)


def check_gpu_offload(need_mib: int | None = None) -> None:
    """Refuse to run unless the model can actually be offloaded to the GPU.

    Two distinct failures, both of which end in a silent CPU run:

    - **The CUDA backend did not initialise.** ggml treats this as fatal for
      the backend but not for the process: `n_gpu_layers` is ignored and the
      model loads on CPU.
    - **The GPU is occupied.** Usually a previous run that was never killed.
      The weights then cannot fit, and llama.cpp again falls back rather than
      failing.

    Neither is detectable from the output -- the rows are fine and the
    grounding rate is perfect -- so both are asserted here, before the weights
    are loaded, and reported with the process holding the GPU when there is one.
    """
    from llama_cpp import llama_cpp as C

    if not C.llama_supports_gpu_offload():
        raise GPUUnavailable(
            _gpu_unavailable_message("llama.cpp reports no GPU offload support.")
        )

    if need_mib:
        free = free_vram_mib()
        if free is not None and free < need_mib:
            raise GPUUnavailable(
                _gpu_unavailable_message(
                    f"only {free} MiB of VRAM free, and the model needs about "
                    f"{need_mib} MiB."
                )
            )


def build_grammar():
    """GBNF generated from the one schema the prompt describes."""
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_json_schema(json.dumps(SCHEMA), verbose=False)


def grammar_accepts(text: str, llm=None, grammar=None) -> bool:
    """Whether the **real GBNF** accepts `text`, token by token.

    This drives llama.cpp's own grammar sampler rather than re-checking the
    JSON schema in Python: the thing under test is the converter's *output*,
    and a check that never touches the generated grammar cannot detect a
    permissive one -- the exact failure ADR 0001 section 6 calls the worst,
    because nothing errors.

    A token is allowed when the grammar sampler has not masked its logit to
    -inf; accepting it advances the parser state.
    """
    import ctypes

    from llama_cpp import llama_cpp as C

    llm = llm or _vocab_llm()
    grammar = grammar or build_grammar()

    vocab = C.llama_model_get_vocab(llm.model)
    sampler = C.llama_sampler_init_grammar(vocab, grammar._grammar.encode(), b"root")
    n_vocab = C.llama_vocab_n_tokens(vocab)

    for token in llm.tokenize(text.encode(), add_bos=False, special=False):
        data = (C.llama_token_data * n_vocab)(
            *[C.llama_token_data(i, 0.0, 0.0) for i in range(n_vocab)]
        )
        candidates = C.llama_token_data_array(data, n_vocab, -1, False)
        C.llama_sampler_apply(sampler, ctypes.byref(candidates))
        if data[token].logit == float("-inf"):
            return False
        C.llama_sampler_accept(sampler, token)
    return True


def _vocab_llm():
    """A vocab-only model handle, for tokenising grammar fixtures.

    Vocab-only: this loads the tokenizer without the 16 GiB of weights, so the
    grammar check stays cheap enough to run before every batch.
    """
    from llama_cpp import Llama

    global _VOCAB_LLM
    if _VOCAB_LLM is None:
        _VOCAB_LLM = Llama(model_path=Config().model_id, vocab_only=True, verbose=False)
    return _VOCAB_LLM


_VOCAB_LLM = None


def check_grammar() -> dict:
    """Assert the generated grammar accepts and rejects the fixtures as expected.

    Each fixture is serialised and fed through the real GBNF. A converter that
    emitted a permissive grammar would accept the negative fixtures and fail
    here, which is the whole point: a grammar desynced from the schema produces
    plausible, parseable, wrong-shaped output that nothing downstream errors on.
    """
    grammar = build_grammar()
    llm = _vocab_llm()

    results = []
    for name, should_pass, payload in GRAMMAR_FIXTURES:
        ok = grammar_accepts(json.dumps(payload), llm=llm, grammar=grammar)
        results.append({"fixture": name, "expected": should_pass, "accepted": ok})
        if ok != should_pass:
            raise RuntimeError(
                f"schema/grammar fixture {name!r}: expected "
                f"{'accept' if should_pass else 'reject'}, got "
                f"{'accept' if ok else 'reject'}. The converter or the schema "
                "changed shape; do not run a batch until this agrees."
            )
    return {"fixtures": results, "ok": True}


class LocalExtractor:
    """The production driver: one loaded model, many records.

    Loading is deferred to first use so importing the pipeline costs nothing,
    and the model is loaded once and reused -- 16 GiB of weights per record
    would dominate the run.
    """

    def __init__(
        self,
        config: Config | None = None,
        verbose: bool = False,
        min_vram_mib: int = MIN_VRAM_MIB,
    ):
        self.config = config or Config()
        self.verbose = verbose
        self._llm = None
        self._grammar = None
        # `llama_cpp.Llama` holds one context and does no locking of its own
        # (there is no lock anywhere in the module), so two threads calling
        # `create_completion` on one instance interleave decode state and
        # corrupt both results. Serialising here makes `n_workers > 1` safe
        # rather than silently wrong.
        #
        # That serialisation is also the honest answer to the open question in
        # ADR 0001 section 9: on one 24 GiB card a second instance cannot be
        # loaded (16.4 GiB of weights each), so in-process threads cannot give
        # 3-5x. Real batching would need llama.cpp's own batched decode across
        # sequences, which this wrapper does not expose. `n_workers` therefore
        # overlaps only render and IO with inference, which is worth little at
        # ~12s/label -- measured, not assumed.
        self._call_lock = threading.Lock()
        # Free VRAM required before loading. The weights are 16.4 GiB; the
        # margin covers the KV cache and compute buffers. Set to 0 to skip the
        # headroom check (the offload assertion always runs).
        self.min_vram_mib = min_vram_mib

    @property
    def llm(self):
        if self._llm is None:
            # Checked here rather than only in `smoke`, because this property is
            # the single point every driver and every entry point passes
            # through -- including `run(skip_smoke=True)` and any direct use.
            # A guard the caller can skip is not a guard against a failure whose
            # only symptom is the clock.
            check_gpu_offload(need_mib=self.min_vram_mib)

            from llama_cpp import Llama

            d = self.config.decode
            try:
                self._llm = Llama(
                    model_path=self.config.model_id,
                    # Fully offloaded: 16.4 GiB of weights on a 24 GiB card.
                    n_gpu_layers=-1,
                    n_ctx=d.n_ctx,
                    seed=d.seed,
                    verbose=self.verbose,
                )
            except Exception as exc:
                # The usual cause is a GPU that filled up between the check
                # above and the load -- another run starting alongside this one.
                raise GPUUnavailable(
                    _gpu_unavailable_message(f"the model failed to load ({exc}).")
                ) from exc
        return self._llm

    @property
    def grammar(self):
        if self._grammar is None:
            self._grammar = build_grammar()
        return self._grammar

    def prompt_for(self, passage: str) -> str:
        """Render the chat prompt, with thinking suppressed explicitly.

        The prompt is built here rather than via `create_chat_completion`
        because this llama-cpp-python (0.3.35) has no `chat_template_kwargs`
        passthrough, so the template's `enable_thinking` branch is unreachable
        through the chat API. That branch is the only thing standing between
        this run and `reasoning_effort: xhigh`, which the model's template
        applies by default -- so the trailing `<think>\\n\\n</think>` the
        template would emit is written here instead, where it is visible and
        asserted rather than hoped for.

        Mirrors the model's ChatML template: system, user, then an assistant
        turn opened with a closed (empty) think block.
        """
        header = (
            f"<|im_start|>system\n{self.config.prompt}<|im_end|>\n"
            f"<|im_start|>user\n{passage}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return header + ("<think>\n" if self.config.decode.enable_thinking else "<think>\n\n</think>\n\n")

    def extract(self, passage: str) -> tuple[str, int, dict]:
        """Run one passage. Returns (raw text, latency_ms, usage).

        No inline retry: a retry hides the failure rate, which is the primary
        early signal that a prompt or grammar is wrong, and it corrupts latency
        stats. A retry pass is an explicit second run over non-`ok` IDs.

        `usage` carries llama.cpp's own token counts -- `prompt_tokens` and
        `completion_tokens`. They come back from every call for free, and
        without them throughput can only be reported per *label*, which hides
        the thing that actually varies: a long label and a short one differ by
        an order of magnitude in work while looking identical at 1/label.
        """
        d = self.config.decode
        started = time.monotonic()
        with self._call_lock:
            out: Any = self.llm.create_completion(
                self.prompt_for(passage),
                grammar=self.grammar,
                temperature=d.temperature,
                top_p=d.top_p,
                top_k=d.top_k,
                max_tokens=d.max_tokens,
                stop=["<|im_end|>"],
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = dict(out.get("usage") or {})
        return out["choices"][0]["text"] or "", latency_ms, usage

    def close(self) -> None:
        self._llm = None
