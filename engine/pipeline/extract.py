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
import threading
import time
from typing import Any

from engine.pipeline.config import Config
from engine.services.extraction.label_purpose_risk import SCHEMA

# ggml's CUDA backend allocates pinned host memory (`cudaHostAlloc`) during
# init. That call is refused in this environment, and ggml treats the failure
# as fatal for the whole backend: it prints `ggml_cuda_init: failed to
# initialize CUDA: initialization error`, reports
# `llama_supports_gpu_offload() == False`, and then **silently ignores
# `n_gpu_layers` and runs on CPU** -- measured here at ~100x slower, which
# turns a 14-day corpus run into an impossible one while looking like it works.
#
# Setting this before the library loads makes ggml skip pinned allocation and
# initialise normally (verified: offload True, 24 GiB visible). It must be set
# before the first llama_cpp import, so it lives at module scope rather than in
# a shell profile where it would be one forgotten export away from a CPU run.
os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")

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


def check_gpu_offload() -> None:
    """Assert the CUDA backend came up, before a batch silently runs on CPU.

    ggml does not fail loudly when its CUDA backend cannot initialise: it
    ignores `n_gpu_layers` and runs on CPU, roughly 100x slower. Nothing in the
    output distinguishes that from a correct run -- the rows are fine, the
    grounding rate is fine, and only the clock says anything is wrong -- so it
    is checked rather than assumed, the same way grounding is.
    """
    from llama_cpp import llama_cpp as C

    if not C.llama_supports_gpu_offload():
        raise RuntimeError(
            "llama.cpp reports no GPU offload support, so a run would fall "
            "back to CPU (~100x slower) without erroring. Check "
            "GGML_CUDA_NO_PINNED=1 is set before llama_cpp is imported, and "
            "that libggml-cuda.so can initialise on this machine."
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

    def __init__(self, config: Config | None = None, verbose: bool = False):
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

    @property
    def llm(self):
        if self._llm is None:
            from llama_cpp import Llama

            d = self.config.decode
            self._llm = Llama(
                model_path=self.config.model_id,
                # Fully offloaded: 16.4 GiB of weights on a 24 GiB card.
                n_gpu_layers=-1,
                n_ctx=d.n_ctx,
                seed=d.seed,
                verbose=self.verbose,
            )
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

    def extract(self, passage: str) -> tuple[str, int]:
        """Run one passage. Returns (raw text, latency_ms).

        No inline retry: a retry hides the failure rate, which is the primary
        early signal that a prompt or grammar is wrong, and it corrupts latency
        stats. A retry pass is an explicit second run over non-`ok` IDs.
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
        return out["choices"][0]["text"] or "", latency_ms

    def close(self) -> None:
        self._llm = None
