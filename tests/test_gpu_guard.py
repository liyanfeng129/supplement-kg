"""The pipeline is GPU-only, and must say so rather than quietly use the CPU.

llama.cpp does not fail when it cannot offload: it ignores `n_gpu_layers` and
runs on CPU at roughly 100x slower. Nothing in the output gives that away --
the rows are right, the grounding rate is perfect, and only the clock differs,
so a corpus run would look healthy while taking years. These tests pin the
refusal and, just as importantly, the diagnosis: the usual cause is a previous
run that was never killed, and the message has to name it.
"""

from __future__ import annotations

import pytest

from engine.pipeline import extract as ex
from engine.pipeline.extract import GPUUnavailable, check_gpu_offload


@pytest.fixture
def no_holders(monkeypatch):
    monkeypatch.setattr(ex, "gpu_holders", lambda: [])


def test_it_refuses_when_llama_cannot_offload(monkeypatch, no_holders):
    monkeypatch.setattr(ex, "free_vram_mib", lambda: 24_000)
    monkeypatch.setattr(
        "llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: False
    )

    with pytest.raises(GPUUnavailable, match="no GPU offload support"):
        check_gpu_offload()


def test_it_refuses_when_the_gpu_has_no_room_left(monkeypatch, no_holders):
    """An occupied GPU is the common case, and CPU fallback is not acceptable."""
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(ex, "free_vram_mib", lambda: 6_690)

    with pytest.raises(GPUUnavailable, match="6690 MiB of VRAM free"):
        check_gpu_offload(need_mib=18_000)


def test_the_refusal_names_the_process_holding_the_gpu(monkeypatch):
    """A leftover run is fixed in one command -- if the message identifies it."""
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(ex, "free_vram_mib", lambda: 2_000)
    monkeypatch.setattr(
        ex,
        "gpu_holders",
        lambda: [{"pid": "12345", "name": "python", "used_mib": "17148"}],
    )

    with pytest.raises(GPUUnavailable) as caught:
        check_gpu_offload(need_mib=18_000)

    message = str(caught.value)
    assert "PID 12345" in message
    assert "kill 12345" in message
    assert "never killed" in message


def test_it_points_at_the_cuda_init_flag_when_nothing_holds_the_gpu(
    monkeypatch, no_holders
):
    """No holder means this is not contention -- it is the backend itself."""
    monkeypatch.setattr(
        "llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: False
    )
    monkeypatch.setattr(ex, "free_vram_mib", lambda: 24_000)

    with pytest.raises(GPUUnavailable, match="GGML_CUDA_NO_PINNED"):
        check_gpu_offload()


def test_it_proceeds_when_the_gpu_is_free(monkeypatch, no_holders):
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(ex, "free_vram_mib", lambda: 24_000)

    check_gpu_offload(need_mib=18_000)  # must not raise


def test_an_unreadable_vram_figure_does_not_block_a_run(monkeypatch, no_holders):
    """nvidia-smi missing is not evidence of a busy GPU; the offload check stands."""
    monkeypatch.setattr("llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(ex, "free_vram_mib", lambda: None)

    check_gpu_offload(need_mib=18_000)  # must not raise


def test_loading_the_model_is_guarded_even_when_smoke_is_skipped(monkeypatch):
    """`run(skip_smoke=True)` must not be a way around the guard."""
    monkeypatch.setattr(
        "llama_cpp.llama_cpp.llama_supports_gpu_offload", lambda: False
    )
    monkeypatch.setattr(ex, "gpu_holders", lambda: [])
    monkeypatch.setattr(ex, "free_vram_mib", lambda: 24_000)

    with pytest.raises(GPUUnavailable):
        ex.LocalExtractor().llm
