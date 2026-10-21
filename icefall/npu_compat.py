# Copyright (c)  2026  Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device / mixed-precision compatibility helpers for the zapformer recipe.

The upstream recipe is written against ``torch.cuda``.  This module exposes a
small, device agnostic API so that the same code runs on Ascend NPUs
(``torch_npu``), CUDA GPUs, or plain CPU.  On an NPU machine ``torch_npu`` is
imported first, which registers the ``npu`` device with PyTorch.
"""

import torch

try:  # Importing torch_npu registers the ``npu`` device type.
    import torch_npu  # noqa: F401

    _HAS_NPU = bool(torch.npu.is_available())
except Exception:  # pragma: no cover - torch_npu not installed / no device
    torch_npu = None
    _HAS_NPU = False

_HAS_CUDA = torch.cuda.is_available()


def is_npu_available() -> bool:
    return _HAS_NPU


def is_cuda_available() -> bool:
    return _HAS_CUDA


def has_accelerator() -> bool:
    return _HAS_NPU or _HAS_CUDA


def device_type() -> str:
    if _HAS_NPU:
        return "npu"
    if _HAS_CUDA:
        return "cuda"
    return "cpu"


def device_count() -> int:
    if _HAS_NPU:
        return torch.npu.device_count()
    if _HAS_CUDA:
        return torch.cuda.device_count()
    return 0


def set_device(rank: int) -> None:
    if _HAS_NPU:
        torch.npu.set_device(rank)
    elif _HAS_CUDA:
        torch.cuda.set_device(rank)


def get_device(rank=None):
    """Return a ``torch.device`` for the given rank (or the current device)."""
    if _HAS_NPU:
        idx = torch.npu.current_device() if rank is None else rank
        return torch.device("npu", idx)
    if _HAS_CUDA:
        idx = torch.cuda.current_device() if rank is None else rank
        return torch.device("cuda", idx)
    return torch.device("cpu")


def max_memory_allocated() -> int:
    if _HAS_NPU:
        return torch.npu.max_memory_allocated()
    if _HAS_CUDA:
        return torch.cuda.max_memory_allocated()
    return 0


def is_bf16_supported() -> bool:
    if _HAS_NPU:
        return True
    if _HAS_CUDA:
        return torch.cuda.is_bf16_supported()
    return True


def get_rng_state():
    if _HAS_NPU:
        return torch.npu.get_rng_state()
    if _HAS_CUDA:
        return torch.cuda.get_rng_state()
    return torch.get_rng_state()


def dist_backend() -> str:
    """The collective-communication backend matching the active device."""
    if _HAS_NPU:
        return "hccl"
    if _HAS_CUDA:
        return "nccl"
    return "gloo"


# ---------------------------------------------------------------------------
# Mixed precision (AMP)
# ---------------------------------------------------------------------------
if _HAS_NPU:
    from torch_npu.npu.amp import GradScaler  # noqa: F401

    def custom_fwd(fwd=None, *, cast_inputs=None, **kwargs):
        import functools as _ft

        fn = _ft.partial(
            torch.amp.custom_fwd, device_type="npu", cast_inputs=cast_inputs
        )
        return fn if fwd is None else fn(fwd)

    def custom_bwd(bwd=None, **kwargs):
        import functools as _ft

        fn = _ft.partial(torch.amp.custom_bwd, device_type="npu")
        return fn if bwd is None else fn(bwd)

    def autocast(enabled: bool = True, dtype=None):
        if dtype is None:
            dtype = torch.float16
        return torch_npu.npu.amp.autocast(enabled=enabled, dtype=dtype)

    def is_autocast_enabled() -> bool:
        return torch.npu.is_autocast_enabled()

    def get_autocast_dtype():
        try:
            return torch.npu.get_autocast_dtype()
        except Exception:  # pragma: no cover
            return torch.float16

else:
    from torch.cuda.amp import GradScaler  # noqa: F401
    from torch.cuda.amp import custom_bwd, custom_fwd  # noqa: F401

    def autocast(enabled: bool = True, dtype=None):
        if dtype is None:
            dtype = torch.float16
        if _HAS_CUDA:
            return torch.amp.autocast(device_type="cuda", enabled=enabled, dtype=dtype)
        return torch.amp.autocast(device_type="cpu", enabled=False)

    def is_autocast_enabled() -> bool:
        return torch.is_autocast_enabled()

    def get_autocast_dtype():
        try:
            return torch.get_autocast_gpu_dtype()
        except Exception:  # pragma: no cover
            return torch.float16


# ---------------------------------------------------------------------------
# torch_npu operator workarounds
# ---------------------------------------------------------------------------
# Bug: on Ascend, ``torch.nn.utils.rnn.pack_padded_sequence`` ignores the
# ``lengths`` argument and packs the *whole padded* input: for a (B, T, C)
# tensor it returns ``data`` with B*T rows instead of ``sum(lengths)`` (the
# batch_sizes / sorted_indices / unsorted_indices metadata are correct, so
# the corruption is silent).  Every transducer decoding function in the
# recipes (zapformer/beam_search.py alone has 8 call sites, including
# ``greedy_search_batch``) consumes ``packed.data`` frame by frame, so on NPU
# the joiner is fed garbage padding frames once batch_size > 1 -- mid
# utterance derailment and truncated hypotheses.
#
# Verified on torch 2.12.0 + torch_npu 2.12.0.post2 (CANN 9.x, 910B4):
#   pack_padded_sequence(x_npu, lengths, batch_first=True, enforce_sorted=False)
#   -> data rows == B*T (wrong); the same call on CPU -> sum(lengths) (right).
#
# Workaround: a pure-torch reimplementation (gather-based, mathematically
# identical to ATen) installed for NPU inputs only; CPU/CUDA inputs keep the
# original implementation.

_ORIG_PACK_PADDED_SEQUENCE = None
_TORCH_FIXES_INSTALLED = False


def _pack_padded_sequence_torch(
    input, lengths, batch_first=True, enforce_sorted=True
):
    """Pure-torch ``pack_padded_sequence`` (same contract as ATen).

    Only used for tensors on the NPU device; see the comment above.
    """
    from torch.nn.utils.rnn import PackedSequence

    if not batch_first:
        input = input.transpose(0, 1)
    B, T = input.shape[0], input.shape[1]

    lengths_c = torch.as_tensor(lengths, dtype=torch.int64).cpu().reshape(-1)
    if lengths_c.numel() == 1 and B > 1:
        # ATen expands a scalar length to the batch size.
        lengths_c = lengths_c.expand(B).contiguous()
    if lengths_c.numel() != B:
        raise ValueError(
            f"lengths must be of size batch size {B}, got {lengths_c.numel()}"
        )
    if B > 0 and int(lengths_c.max()) > T:
        raise ValueError(
            f"max(lengths)={int(lengths_c.max())} exceeds input length T={T}"
        )

    if enforce_sorted:
        sorted_indices = None
        unsorted_indices = None
        sorted_lengths = lengths_c
        order = None
    else:
        # ATen sorts lengths in descending order (stable for ties).
        order = torch.argsort(lengths_c, descending=True, stable=True)
        sorted_indices = order
        unsorted_indices = torch.argsort(order)
        sorted_lengths = lengths_c[order]

    total = int(sorted_lengths.sum()) if B > 0 else 0
    max_len = int(sorted_lengths.max()) if B > 0 else 0
    if max_len > 0:
        batch_sizes = (
            sorted_lengths.view(1, -1) > torch.arange(max_len).view(-1, 1)
        ).sum(dim=1)
    else:
        batch_sizes = torch.zeros(0, dtype=torch.int64)

    if total > 0:
        # ``PackedSequence.data`` is time-major: grouped by timestep t, and
        # within each t by (sorted) sequence index.  ``active[t, i]`` marks
        # that sequence i (in sorted order) still has a frame at time t.
        active = (
            sorted_lengths.view(1, -1) > torch.arange(max_len).view(-1, 1)
        )  # (max_len, B) bool
        pos = active.nonzero()  # (total, 2), rows ordered by t then i
        t = pos[:, 0]
        row = pos[:, 1]
        if order is not None:
            row = order[row]
        data = input[row.to(input.device), t.to(input.device)]
    else:
        data = input.new_empty((0,) + tuple(input.shape[2:]))

    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


def install_torch_fixes() -> bool:
    """Install NPU workarounds for broken torch_npu operator implementations.

    Currently this replaces ``torch.nn.utils.rnn.pack_padded_sequence`` with
    a wrapper that routes *NPU* inputs through the pure-torch implementation
    (:func:`_pack_padded_sequence_torch`); tensors on any other device use
    the original PyTorch code path.  The function is idempotent and is also
    called automatically when this module is imported on an NPU machine.

    Returns:
      True if the fixes are (already) installed, False if they do not apply
      (no NPU present).
    """
    global _ORIG_PACK_PADDED_SEQUENCE, _TORCH_FIXES_INSTALLED
    if _TORCH_FIXES_INSTALLED:
        return True
    if not _HAS_NPU:
        return False

    import torch.nn.utils.rnn as _rnn

    _ORIG_PACK_PADDED_SEQUENCE = _rnn.pack_padded_sequence

    def _pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=True, **kwargs):
        if isinstance(input, torch.Tensor) and input.device.type == "npu":
            return _pack_padded_sequence_torch(
                input, lengths, batch_first=batch_first, enforce_sorted=enforce_sorted
            )
        return _ORIG_PACK_PADDED_SEQUENCE(
            input,
            lengths,
            batch_first=batch_first,
            enforce_sorted=enforce_sorted,
            **kwargs,
        )

    _pack_padded_sequence.__doc__ = (
        "NPU-aware wrapper around torch.nn.utils.rnn.pack_padded_sequence "
        "(see icefall.npu_compat.install_torch_fixes)."
    )
    _rnn.pack_padded_sequence = _pack_padded_sequence
    _TORCH_FIXES_INSTALLED = True
    return True


# Self-install on import on NPU machines so that every consumer of the
# compat layer gets the fixes without an explicit call.
install_torch_fixes()
