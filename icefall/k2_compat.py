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

"""A minimal, Ascend-NPU friendly stand-in for the ``k2`` package.

``k2`` ships operator kernels for CPU and CUDA only, so it cannot be used on
Ascend NPUs.  Only the small part of the ``k2`` API that is exercised by the
training / fine-tuning path is required, so we provide it here:

  * :class:`RaggedTensor` -- enough of the ragged-tensor API for the RNN-T
    training path (``pad``, ``to``, ``values``, ``shape.row_splits`` ...).
  * ``k2.ragged.cat`` -- used by ``icefall.utils.concat`` / ``add_sos``.
  * the RNN-T losses (see :mod:`icefall._rnnt_loss_torch`).
  * ``k2.version`` -- used by ``icefall.env.get_env_info``.

If the real ``k2`` is installed, it is used instead and this module is never
imported (see ``icefall/__init__.py``).
"""

import types
from typing import List, Optional, Union

import torch
from torch import Tensor

from ._rnnt_loss_torch import (  # noqa: F401
    do_rnnt_pruning,
    get_rnnt_prune_ranges,
    mutual_information_recursion,
    rnnt_loss,
    rnnt_loss_pruned,
    rnnt_loss_simple,
    rnnt_loss_smoothed,
)

with_cuda = False

# ---------------------------------------------------------------------------
# version (k2.version is imported by icefall.env)
# ---------------------------------------------------------------------------
version = types.ModuleType("k2.version")
version.__version__ = "0.0.0-torch-fallback"
version.__build_type__ = "torch-fallback (no k2, NPU friendly)"
version.__git_sha1__ = "unknown"
version.__git_date__ = "unknown"


# ---------------------------------------------------------------------------
# RaggedTensor
# ---------------------------------------------------------------------------
class _RaggedShape:
    """The subset of ``k2.RaggedShape`` that we need."""

    def __init__(self, lengths: Tensor, num_axes: int = 2):
        # ``lengths`` is a 1-D tensor with the size of each sub-list.
        self._lengths = lengths
        self._num_axes = num_axes

    @property
    def num_axes(self) -> int:
        return self._num_axes

    def row_splits(self, axis: int, cache: bool = False) -> Tensor:
        """Return the exclusive cumsum of the sub-list sizes."""
        if axis == 0:
            n = self._lengths.shape[0]
            return torch.arange(n + 1, dtype=torch.int64, device=self._lengths.device)
        assert axis == 1, axis
        lengths = self._lengths.to(torch.int64)
        zero = torch.zeros(1, dtype=torch.int64, device=lengths.device)
        return torch.cat((zero, torch.cumsum(lengths, dim=0)), dim=0)

    def tot_size(self, axis: int) -> int:
        if axis == 0:
            return int(self._lengths.shape[0])
        return int(self._lengths.sum().item())


class RaggedTensor:
    """A minimal ragged tensor with two axes ``[utt][label]``.

    Only the operations used by the RNN-T training path are implemented.
    """

    def __init__(
        self,
        data: Optional[Union[List[List[int]], List[Tensor], Tensor]] = None,
        values: Optional[Tensor] = None,
        shape=None,
        dtype: Optional[torch.dtype] = None,
        device=None,
    ):
        if data is not None and values is None:
            if torch.is_tensor(data):
                # A (possibly padded) 2-D tensor: each row is a sub-list.
                if data.dim() == 1:
                    lengths = torch.tensor([data.numel()], dtype=torch.int64)
                    flat = data.reshape(-1).to(torch.int64)
                else:
                    lengths = torch.full(
                        (data.shape[0],), data.shape[1], dtype=torch.int64
                    )
                    flat = data.reshape(-1).to(torch.int64)
            elif isinstance(data, (list, tuple)) and (
                len(data) == 0 or not torch.is_tensor(data[0])
            ):
                lengths = torch.tensor([len(x) for x in data], dtype=torch.int64)
                flat = torch.tensor(
                    [int(v) for x in data for v in x], dtype=torch.int64
                )
            else:
                # A list of 1-D tensors.
                lengths = torch.tensor([int(x.numel()) for x in data], dtype=torch.int64)
                flat = torch.cat([x.reshape(-1).to(torch.int64) for x in data], dim=0)
            if dtype is not None:
                flat = flat.to(dtype)
            lengths = lengths.to(flat.device)
        else:
            # Constructed from ``values`` (and optionally a ``shape``).
            assert values is not None, "RaggedTensor requires data or values"
            flat = values.reshape(-1)
            n = int(shape._lengths.shape[0]) if shape is not None else 1
            lengths = torch.tensor([flat.numel() // max(n, 1)] * n, dtype=torch.int64)

        if device is not None:
            flat = flat.to(device)
            lengths = lengths.to(device)

        self._values = flat
        self._lengths = lengths
        self.dtype = flat.dtype
        self._shape = _RaggedShape(lengths)

    # -- basic properties ---------------------------------------------------
    @property
    def num_axes(self) -> int:
        return 2

    @property
    def dim0(self) -> int:
        return int(self._lengths.shape[0])

    @property
    def device(self):
        return self._values.device

    @property
    def values(self) -> Tensor:
        return self._values

    @property
    def shape(self) -> _RaggedShape:
        return self._shape

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        out = []
        offset = 0
        for length in self._lengths.tolist():
            out.append(self._values[offset : offset + length].tolist())
            offset += length
        return f"RaggedTensor({out})"

    # -- API used by the training path -------------------------------------
    def tot_size(self, axis: int) -> int:
        return self._shape.tot_size(axis)

    def to(self, device=None, dtype=None) -> "RaggedTensor":
        if device is None and dtype is None:
            return self
        new = RaggedTensor.__new__(RaggedTensor)
        new._values = self._values.to(device=device, dtype=dtype)
        new._lengths = self._lengths.to(device=device)
        new.dtype = new._values.dtype
        new._shape = _RaggedShape(new._lengths)
        return new

    def pad(self, mode: str = "constant", padding_value: int = 0) -> Tensor:
        """Return a 2-D padded tensor of shape (dim0, max_len)."""
        assert mode == "constant", mode
        lengths = self._lengths
        n = int(lengths.shape[0])
        if n == 0:
            return self._values.new_empty((0, 0))
        max_len = int(lengths.max().item())
        out = torch.full(
            (n, max_len),
            int(padding_value),
            dtype=self._values.dtype,
            device=self._values.device,
        )
        if max_len > 0:
            lengths_cpu = lengths.to(torch.int64).cpu()
            offsets = torch.cumsum(lengths_cpu, dim=0) - lengths_cpu
            rows = torch.repeat_interleave(
                torch.arange(n, dtype=torch.int64), lengths_cpu
            )
            cols = torch.arange(int(lengths_cpu.sum()), dtype=torch.int64) - offsets[
                rows
            ]
            out[rows.to(out.device), cols.to(out.device)] = self._values
        return out


# ---------------------------------------------------------------------------
# k2.ragged sub-module (only ``cat`` is needed for the training path)
# ---------------------------------------------------------------------------
def _cat(raggeds: List[RaggedTensor], axis: int = 1) -> RaggedTensor:
    assert axis == 1, f"only axis=1 is supported, got {axis}"
    assert len(raggeds) >= 1
    n = raggeds[0].dim0
    for r in raggeds:
        assert r.dim0 == n, (r.dim0, n)
    lengths = torch.stack([r._lengths for r in raggeds], dim=0).sum(dim=0)
    device = raggeds[0].device
    dtype = raggeds[0].dtype
    # Simple, readable implementation (this is used once per batch, on the
    # symbol sequence, so it is not performance critical).
    rows = [r._values.tolist() for r in raggeds]
    lens = [r._lengths.tolist() for r in raggeds]
    flat = []
    for b in range(n):
        for r in range(len(raggeds)):
            flat.extend(rows[r][sum(lens[r][:b]) : sum(lens[r][: b + 1])])
    out = torch.tensor(flat, dtype=dtype, device=device)
    new = RaggedTensor.__new__(RaggedTensor)
    new._values = out
    new._lengths = lengths.to(device)
    new.dtype = dtype
    new._shape = _RaggedShape(new._lengths)
    return new


ragged = types.ModuleType("k2.ragged")
ragged.cat = _cat


class SymbolTable:
    """Minimal torch-only replacement for ``k2.SymbolTable``.

    Covers what the recipes use: :meth:`from_file` (icefall's tokens.txt
    format, one ``symbol id`` pair per line), ``table["sym"]`` /
    ``table[id]`` lookups, ``table.symbols`` (used by
    ``icefall.utils.num_tokens``) and ``len(table)``.
    """

    def __init__(self, id2sym):
        self._id2sym = dict(id2sym)
        self._sym2id = {v: k for k, v in self._id2sym.items()}

    @classmethod
    def from_file(cls, filename: str) -> "SymbolTable":
        id2sym = {}
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                sym, idx = line.rsplit(maxsplit=1)
                id2sym[int(idx)] = sym
        return cls(id2sym)

    @property
    def symbols(self):
        return [self._id2sym[i] for i in sorted(self._id2sym)]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._id2sym[key]
        return self._sym2id[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self):
        return len(self._id2sym)

    def __contains__(self, key):
        return key in self._id2sym or key in self._sym2id

    def __repr__(self):
        return f"SymbolTable({len(self._id2sym)} symbols)"


class _Unavailable:
    """Placeholder for the (large) part of the k2 API we do not implement.

    Referencing the name is fine (it is used in type annotations all over
    icefall); using it raises a clear error.
    """

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"k2.{self._name} is not available: this is the torch-based k2 "
            f"replacement used on Ascend NPU, which only implements the RNN-T "
            f"training path. Install a CPU/CUDA build of k2 for FSA/decoding ops."
        )

    def __repr__(self) -> str:
        return f"<k2.{self._name} (unavailable in the torch k2 shim)>"


def __getattr__(name: str):
    """PEP-562 module level fallback (see :class:`_Unavailable`)."""
    if name.startswith("__"):
        raise AttributeError(name)
    return _Unavailable(name)



def install() -> None:
    """Register this module (and its sub-modules) as ``k2`` in ``sys.modules``.

    This is only called when the real ``k2`` package is not available.
    """
    import sys

    sys.modules["k2"] = sys.modules[__name__]
    sys.modules["k2.version"] = version
    sys.modules["k2.ragged"] = ragged

