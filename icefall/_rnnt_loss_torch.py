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

"""Pure-PyTorch / Ascend-NPU friendly implementation of the RNN-T (transducer)
losses that are normally provided by ``k2``.

Only the API surface used by the training / fine-tuning path of the zapformer
recipe is provided:

  * :func:`mutual_information_recursion`
  * :func:`rnnt_loss_simple`
  * :func:`rnnt_loss`
  * :func:`rnnt_loss_pruned`
  * :func:`rnnt_loss_smoothed`
  * :func:`get_rnnt_prune_ranges`
  * :func:`do_rnnt_pruning`

The log-probability construction (``get_rnnt_logprobs*``) and the pruning range
computation are a faithful PyTorch port of the corresponding functions in the
``k2`` project (Apache-2.0), because they are already written in terms of plain
``torch`` ops.

The numerical core -- :func:`mutual_information_recursion` -- is re-implemented
here using only ops that are supported on the Ascend NPU.  It uses the same
recursion as ``k2``::

    p[b, 0, 0] = 0
    p[b, s, t] = log_add(p[b, s-1, t] + px[b, s-1, t],
                         p[b, s, t-1] + py[b, s, t-1])

The recursion is evaluated in a loop over the time axis ``t`` while the symbol
axis ``s`` is resolved in closed form with a numerically stable cumulative
log-sum-exp (implemented on top of ``cummax``/``cumsum`` so that it does not
rely on ``torch.logcumsumexp``, which currently falls back to the CPU on NPU).
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------
def _logcumsumexp(v: Tensor, dim: int = -1) -> Tensor:
    """A ``torch.logcumsumexp`` that is implemented with NPU-supported ops.

    ``torch.logcumsumexp`` currently falls back to the CPU on Ascend NPU, so we
    implement the inclusive scan with the associative ``logaddexp`` operator
    (Hillis-Steele), which only uses ``logaddexp`` and slicing.

    The result is mathematically exact; the scan takes ``ceil(log2(n))`` steps.
    """
    n = v.shape[dim]
    if n <= 1:
        return v
    y = v
    off = 1
    while off < n:
        pad_shape = list(y.shape)
        pad_shape[dim] = off
        pad = torch.full(
            pad_shape, float("-inf"), dtype=y.dtype, device=y.device
        )
        y_shift = torch.cat([pad, y.narrow(dim, 0, n - off)], dim=dim)
        y = torch.logaddexp(y, y_shift)
        off *= 2
    return y


# A large, finite stand-in for -infinity inside the recurrence.  Using literal
# -inf would make the backward of ``logaddexp(-inf, -inf)`` produce NaN
# (``sigmoid(-inf - -inf)``), while exp(-1e30) still underflows to 0, so a
# finite sentinel changes nothing in the forward result.
_NEG_BIG = -1.0e30


def _scan_step(a: Tensor, cum: Tensor, neg_inf: float = float("-inf")) -> Tensor:
    """One symbol-axis sweep of the mutual-information recursion.

    Solves, for every batch element, the first-order recurrence in the
    log-semiring::

        q[s] = log_add(q[s-1] + a[s-1], cum[s]),   q[0] = cum[0]

    which has the closed form ``q[s] = A[s] + logcumsumexp(cum - A)[s]`` with
    ``A`` the exclusive prefix sum of ``a``.

    Args:
      a: shape (B, S); the px log-probs of the current frame.
      cum: shape (B, S+1); the "blank/bias" term of the current frame.

    Returns:
      A tensor of shape (B, S+1).
    """
    # ``a`` may contain -inf (no symbol can be emitted on some frames).  Replace
    # it by 0 for batches that are not going to use the result, so that the
    # prefix sum stays finite and never produces NaN.
    a = torch.where(torch.isinf(a) & (a < 0), torch.zeros_like(a), a)
    cum = torch.where(
        torch.isinf(cum) & (cum < 0), torch.full_like(cum, _NEG_BIG), cum
    )
    zeros = torch.zeros_like(cum[:, :1])
    A = torch.cat((zeros, torch.cumsum(a, dim=1)), dim=1)  # (B, S+1)
    c = cum - A
    return A + _logcumsumexp(c, dim=1)


def _mutual_information_forward(
    px: Tensor, py: Tensor, boundary: Optional[Tensor]
) -> Tensor:
    """Differentiable forward pass of the mutual information recursion.

    Args:
      px: shape (B, S, T+1) for the "regular" recursion.
      py: shape (B, S+1, T).
      boundary: optional int64 tensor of shape (B, 4) containing
        ``[s_begin, t_begin, s_end, t_end]``.  Only ``s_begin == t_begin == 0``
        is supported (which is what the RNN-T training path uses).

    Returns:
      A tensor of shape (B,) with ``p[b, s_end, t_end]``.
    """
    B, S, T1 = px.shape
    T = py.shape[-1]
    S1 = S + 1
    device = px.device
    dtype = px.dtype
    neg_inf = float("-inf")

    if boundary is None:
        s_end = torch.full((B,), S, dtype=torch.long, device=device)
        t_end = torch.full((B,), T, dtype=torch.long, device=device)
    else:
        assert boundary.dtype == torch.int64, boundary.dtype
        assert boundary.shape == (B, 4), (boundary.shape, B)
        assert bool((boundary[:, 0] == 0).all()) and bool(
            (boundary[:, 1] == 0).all()
        ), "mutual_information_recursion: only s_begin == t_begin == 0 is supported"
        s_end = boundary[:, 2].to(torch.long)
        t_end = boundary[:, 3].to(torch.long)

    assert bool((s_end >= 0).all()) and bool((s_end <= S).all()), s_end
    assert bool((t_end >= 0).all()) and bool((t_end <= T).all()), t_end

    s_end_idx = s_end.reshape(B, 1)  # (B, 1)

    t_end = t_end.reshape(B, 1)
    ans = torch.full((B,), neg_inf, dtype=dtype, device=device)

    # t = 0: the only non -inf entry of the bias term is at s == 0.
    cum = torch.full((B, S1), neg_inf, dtype=dtype, device=device)
    cum[:, 0] = 0.0
    hit = (t_end == 0).reshape(B)
    ans = torch.where(hit, cum.gather(1, s_end_idx).reshape(B), ans)

    prev = _scan_step(px[:, :, 0], cum)  # q at t == 0

    for t in range(1, T + 1):
        cum = prev + py[:, :, t - 1]  # (B, S+1)
        hit = (t_end == t).reshape(B)
        ans = torch.where(hit, cum.gather(1, s_end_idx).reshape(B), ans)

        if t < T:
            a = px[:, :, t]
            scan = _scan_step(a, cum)
            use_scan = (t_end > t).reshape(B, 1)
            prev = torch.where(use_scan, scan, cum)

    return ans


def mutual_information_recursion(
    px: Tensor,
    py: Tensor,
    boundary: Optional[Tensor] = None,
    return_grad: bool = False,
) -> Union[Tuple[Tensor, Tuple[Tensor, Tensor]], Tensor]:
    """Drop-in replacement for :func:`k2.mutual_information_recursion`.

    Returns the log mutual information of shape ``(B,)``.  If ``return_grad``
    is True, a tuple ``(scores, (px_grad, py_grad))`` is returned, where the
    grads are ``d(sum(scores)) / d(px)`` and ``d(sum(scores)) / d(py)`` (the
    "occupation probabilities" used to compute the pruning ranges).
    """
    assert px.ndim == 3, px.shape
    B, S, T1 = px.shape
    T = py.shape[-1]
    assert px.shape[-1] in [T, T + 1], (px.shape, T)
    assert py.shape == (B, S + 1, T), (py.shape, B, S, T)
    assert px.dtype == py.dtype, (px.dtype, py.dtype)

    px = px.contiguous()
    py = py.contiguous()

    # Build one differentiable forward graph.
    scores = _mutual_information_forward(px, py, boundary)

    px_grad, py_grad = None, None
    if return_grad:
        if px.requires_grad or py.requires_grad:
            # Reuse the same graph (retain_graph) instead of running a second
            # forward pass; this is the "occupation probability" used to
            # compute the pruning ranges.
            (px_grad, py_grad) = torch.autograd.grad(
                scores.sum(), (px, py), retain_graph=True, create_graph=False,
                allow_unused=True,
            )
        else:  # pragma: no cover - degenerate case, k2 still supports it
            with torch.enable_grad():
                px_ = px.detach().requires_grad_(True)
                py_ = py.detach().requires_grad_(True)
                out = _mutual_information_forward(px_, py_, boundary)
                (px_grad, py_grad) = torch.autograd.grad(
                    out.sum(), (px_, py_), retain_graph=False, create_graph=False
                )

    if return_grad:
        return scores, (px_grad, py_grad)
    return scores


# ---------------------------------------------------------------------------
# Ported / re-implemented log-probability and loss functions (k2 API)
# ---------------------------------------------------------------------------
def fix_for_boundary(px: Tensor, boundary: Optional[Tensor] = None) -> Tensor:
    """Insert -inf at the position ``boundary[b, 3]`` of every row of ``px``."""
    if boundary is None:
        return px
    B, S, T1 = px.shape
    boundary = boundary[:, 3].reshape(B, 1, 1).expand(B, S, T1)
    return px.scatter_(dim=2, index=boundary, value=float("-inf"))


def _validate_st_lengths(
    S: int,
    T: int,
    is_rnnt_type_regular: bool,
    boundary: Optional[Tensor] = None,
):
    assert S >= 0, S
    if boundary is None:
        assert (
            is_rnnt_type_regular or T >= S
        ), f"Modified transducer requires T >= S, but got T={T} and S={S}"
    else:
        Ss = boundary[:, 2]
        Ts = boundary[:, 3]
        assert (
            is_rnnt_type_regular or (Ts >= Ss).all()
        ), f"Modified transducer requires T >= S, but got T={Ts} and S={Ss}"


def get_rnnt_logprobs(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    rnnt_type: str = "regular",
    boundary: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    assert lm.ndim == 3, lm.shape
    assert am.ndim == 3, am.shape
    assert lm.shape[0] == am.shape[0], (lm.shape, am.shape)
    assert lm.shape[2] == am.shape[2], (lm.shape, am.shape)

    (B, T, C) = am.shape
    S = lm.shape[1] - 1
    assert symbols.shape == (B, S), (symbols.shape, B, S)
    assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type
    _validate_st_lengths(S, T, rnnt_type == "regular", boundary)

    am_max, _ = torch.max(am, dim=2, keepdim=True)
    lm_max, _ = torch.max(lm, dim=2, keepdim=True)
    am_probs = (am - am_max).exp()
    lm_probs = (lm - lm_max).exp()
    normalizers = (
        torch.matmul(lm_probs, am_probs.transpose(1, 2))
        + torch.finfo(am_probs.dtype).tiny
    ).log()
    normalizers = normalizers + lm_max + am_max.transpose(1, 2)

    px_am = torch.gather(
        am.transpose(1, 2), dim=1, index=symbols.unsqueeze(2).expand(B, S, T)
    )

    if rnnt_type == "regular":
        px_am = torch.cat(
            (
                px_am,
                torch.full(
                    (B, S, 1),
                    float("-inf"),
                    device=px_am.device,
                    dtype=px_am.dtype,
                ),
            ),
            dim=2,
        )

    px_lm = torch.gather(lm[:, :S], dim=2, index=symbols.unsqueeze(-1))
    px = px_am + px_lm
    px[:, :, :T] -= normalizers[:, :S, :]

    py_am = am[:, :, termination_symbol].unsqueeze(1)
    py_lm = lm[:, :, termination_symbol].unsqueeze(2)
    py = py_am + py_lm - normalizers

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px = px + py[:, 1:, :]

    return (px, py)


def rnnt_loss_simple(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
    px, py = get_rnnt_logprobs(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )

    if delay_penalty > 0.0:
        B, S, T0 = px.shape
        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = torch.tensor(
                (T - 1) / 2, dtype=px.dtype, device=px.device
            ).expand(B, 1, 1)
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = offset.reshape(B, 1, 1) - torch.arange(
            T0, device=px.device
        ).reshape(1, 1, T0)
        penalty = penalty * delay_penalty
        px = px + penalty.to(px.dtype)

    scores_and_grads = mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -torch.mean(negated_loss)
    elif reduction == "sum":
        loss = -torch.sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )
    return (loss, scores_and_grads[1]) if return_grad else loss


def get_rnnt_logprobs_joint(
    logits: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    rnnt_type: str = "regular",
    boundary: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    assert logits.ndim == 4, logits.shape
    (B, T, S1, C) = logits.shape
    S = S1 - 1
    assert symbols.shape == (B, S), (symbols.shape, B, S)
    assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type
    _validate_st_lengths(S, T, rnnt_type == "regular", boundary)

    normalizers = torch.logsumexp(logits, dim=3)
    normalizers = normalizers.permute((0, 2, 1))

    px = torch.gather(
        logits, dim=3, index=symbols.reshape(B, 1, S, 1).expand(B, T, S, 1)
    ).squeeze(-1)
    px = px.permute((0, 2, 1))

    if rnnt_type == "regular":
        px = torch.cat(
            (
                px,
                torch.full(
                    (B, S, 1), float("-inf"), device=px.device, dtype=px.dtype
                ),
            ),
            dim=2,
        )

    px[:, :, :T] -= normalizers[:, :S, :]

    py = logits[:, :, :, termination_symbol].permute((0, 2, 1)).clone()
    py -= normalizers

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px = px + py[:, 1:, :]

    return (px, py)


def rnnt_loss(
    logits: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    boundary: Optional[Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
) -> Tensor:
    px, py = get_rnnt_logprobs_joint(
        logits=logits,
        symbols=symbols,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )

    if delay_penalty > 0.0:
        B, S, T0 = px.shape
        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = torch.tensor(
                (T - 1) / 2, dtype=px.dtype, device=px.device
            ).expand(B, 1, 1)
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = offset.reshape(B, 1, 1) - torch.arange(
            T0, device=px.device
        ).reshape(1, 1, T0)
        penalty = penalty * delay_penalty
        px = px + penalty.to(px.dtype)

    negated_loss = mutual_information_recursion(px=px, py=py, boundary=boundary)
    if reduction == "none":
        return -negated_loss
    elif reduction == "mean":
        return -torch.mean(negated_loss)
    elif reduction == "sum":
        return -torch.sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )


def _monotonic_lower_bound(x: torch.Tensor) -> torch.Tensor:
    x = torch.flip(x, dims=(-1,))
    x, _ = torch.cummin(x, dim=-1)
    x = torch.flip(x, dims=(-1,))
    return x


def _adjust_pruning_lower_bound(
    s_begin: torch.Tensor, s_range: int
) -> torch.Tensor:
    (B, T) = s_begin.shape
    s_begin = _monotonic_lower_bound(s_begin)
    s_begin = -(
        s_begin - (s_range - 1) * torch.arange(0, T, device=s_begin.device)
    )
    s_begin = _monotonic_lower_bound(s_begin)
    s_begin = torch.clamp(s_begin, min=0)
    s_begin = -(
        s_begin - (s_range - 1) * torch.arange(0, T, device=s_begin.device)
    )
    return s_begin


def get_rnnt_prune_ranges(
    px_grad: torch.Tensor,
    py_grad: torch.Tensor,
    boundary: torch.Tensor,
    s_range: int,
) -> torch.Tensor:
    (B, S, T1) = px_grad.shape
    T = py_grad.shape[-1]

    is_regular = T1 != T

    assert T1 in [T, T + 1], (T1, T)
    S1 = S + 1
    assert py_grad.shape == (B, S1, T), (py_grad.shape, B, S1, T)
    assert boundary.shape == (B, 4), (boundary.shape, B)

    _validate_st_lengths(S, T, is_regular, boundary)

    if is_regular:
        Ss = boundary[:, 2]
        Ts = boundary[:, 3]
        s_range_min = (
            Ss.sub(1).div(Ts, rounding_mode="trunc").add(2).max().item()
        )
        if s_range < s_range_min:
            print(
                f"Warning: get_rnnt_prune_ranges - got s_range={s_range} "
                f"for boundaries S={Ss}, T={Ts}. Adjusting to {s_range_min}"
            )
            s_range = s_range_min

    if s_range > S:
        print(
            f"Warning: get_rnnt_prune_ranges - got s_range={s_range} "
            f"for boundaries S={S}. Adjusting to {S + 1}"
        )
        s_range = S + 1

    if is_regular:
        assert (
            s_range >= 2
        ), f"""Pruning range for standard RNN-T should be equal to or greater
        than 2, or no valid paths could survive pruning. Given {s_range}"""
    else:
        assert (
            s_range >= 1
        ), f"""Pruning range for modified RNN-T should be equal to or greater
        than 1, or no valid paths could survive pruning. Given {s_range}"""

    (B_stride, S_stride, T_stride) = py_grad.stride()
    blk_grad = torch.as_strided(
        py_grad,
        (B, S1 - s_range + 1, s_range, T),
        (B_stride, S_stride, S_stride, T_stride),
    )
    blk_sum_grad = torch.sum(blk_grad, axis=2)

    px_pad = torch.zeros((B, 1, T1), dtype=px_grad.dtype, device=px_grad.device)
    px_grad_pad = torch.cat((px_pad, px_grad), dim=1)

    final_grad = blk_sum_grad - px_grad_pad[:, : S1 - s_range + 1, :T]

    s_begin = torch.argmax(final_grad, axis=1)

    mask = torch.arange(0, T, device=px_grad.device).reshape(1, T).expand(B, T)
    mask = mask < boundary[:, 3].reshape(B, 1) - 1

    s_begin_padding = boundary[:, 2].reshape(B, 1) - s_range + 1
    s_begin_padding = torch.clamp(s_begin_padding, min=0)

    s_begin = torch.where(mask, s_begin, s_begin_padding)

    s_begin = _adjust_pruning_lower_bound(s_begin, 2 if T1 == T else s_range)

    ranges = s_begin.reshape((B, T, 1)).expand((B, T, s_range)) + torch.arange(
        s_range, device=px_grad.device
    )

    return ranges


def do_rnnt_pruning(
    am: torch.Tensor, lm: torch.Tensor, ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert ranges.shape[0] == am.shape[0], (ranges.shape, am.shape)
    assert ranges.shape[0] == lm.shape[0], (ranges.shape, lm.shape)
    assert am.shape[1] == ranges.shape[1], (am.shape, ranges.shape)
    (B, T, s_range) = ranges.shape
    (B, S1, decoder_dim) = lm.shape
    encoder_dim = am.shape[-1]
    assert am.shape == (B, T, encoder_dim), (am.shape, B, T, encoder_dim)

    am_pruned = am.unsqueeze(2).expand((B, T, s_range, encoder_dim))

    lm_pruned = torch.gather(
        lm,
        dim=1,
        index=ranges.reshape(B, T * s_range, 1).expand(
            (B, T * s_range, decoder_dim)
        ),
    ).reshape(B, T, s_range, decoder_dim)
    return am_pruned, lm_pruned


def _roll_by_shifts(src: torch.Tensor, shifts: torch.LongTensor):
    assert src.dim() == 3, src.shape
    (B, T, S) = src.shape
    assert shifts.shape == (B, T), (shifts.shape, B, T)

    index = (
        torch.arange(S, device=src.device)
        .view((1, S))
        .repeat((T, 1))
        .repeat((B, 1, 1))
    )
    index = (index - shifts.reshape(B, T, 1)) % S
    return torch.gather(src, 2, index)


def get_rnnt_logprobs_pruned(
    logits: Tensor,
    symbols: Tensor,
    ranges: Tensor,
    termination_symbol: int,
    boundary: Tensor,
    rnnt_type: str = "regular",
) -> Tuple[Tensor, Tensor]:
    assert logits.ndim == 4, logits.shape
    (B, T, s_range, C) = logits.shape
    assert ranges.shape == (B, T, s_range), (ranges.shape, B, T, s_range)
    (B, S) = symbols.shape
    assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type
    _validate_st_lengths(S, T, rnnt_type == "regular", boundary)

    normalizers = torch.logsumexp(logits, dim=3)

    symbols_with_terminal = torch.cat(
        (
            symbols,
            torch.tensor(
                [termination_symbol] * B,
                dtype=torch.int64,
                device=symbols.device,
            ).reshape((B, 1)),
        ),
        dim=1,
    )

    pruned_symbols = torch.gather(
        symbols_with_terminal.unsqueeze(1).expand((B, T, S + 1)),
        dim=2,
        index=ranges,
    )

    px = torch.gather(
        logits, dim=3, index=pruned_symbols.reshape(B, T, s_range, 1)
    ).squeeze(-1)
    px = px - normalizers

    px = torch.cat(
        (
            px,
            torch.full(
                (B, T, S + 1 - s_range),
                float("-inf"),
                device=px.device,
                dtype=px.dtype,
            ),
        ),
        dim=2,
    )

    px = _roll_by_shifts(px, ranges[:, :, 0])[:, :, :S]

    px = px.permute((0, 2, 1))

    if rnnt_type == "regular":
        px = torch.cat(
            (
                px,
                torch.full(
                    (B, S, 1), float("-inf"), device=px.device, dtype=px.dtype
                ),
            ),
            dim=2,
        )

    py = logits[:, :, :, termination_symbol].clone()
    py = py - normalizers

    py = torch.cat(
        (
            py,
            torch.full(
                (B, T, S + 1 - s_range),
                float("-inf"),
                device=py.device,
                dtype=py.dtype,
            ),
        ),
        dim=2,
    )

    py = _roll_by_shifts(py, ranges[:, :, 0])
    py = py.permute((0, 2, 1))

    if rnnt_type == "regular":
        px = fix_for_boundary(px, boundary)
    elif rnnt_type == "constrained":
        px = px + py[:, 1:, :]

    return (px, py)


def rnnt_loss_pruned(
    logits: Tensor,
    symbols: Tensor,
    ranges: Tensor,
    termination_symbol: int,
    boundary: Tensor = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
) -> Tensor:
    px, py = get_rnnt_logprobs_pruned(
        logits=logits,
        symbols=symbols,
        ranges=ranges,
        termination_symbol=termination_symbol,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )

    if delay_penalty > 0.0:
        B, S, T0 = px.shape
        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = torch.tensor(
                (T - 1) / 2, dtype=px.dtype, device=px.device
            ).expand(B, 1, 1)
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = offset.reshape(B, 1, 1) - torch.arange(
            T0, device=px.device
        ).reshape(1, 1, T0)
        penalty = penalty * delay_penalty
        px = px + penalty.to(px.dtype)

    negated_loss = mutual_information_recursion(px=px, py=py, boundary=boundary)
    if reduction == "none":
        return -negated_loss
    elif reduction == "mean":
        return -torch.mean(negated_loss)
    elif reduction == "sum":
        return -torch.sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )


def get_rnnt_logprobs_smoothed(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[Tensor] = None,
    rnnt_type: str = "regular",
) -> Tuple[Tensor, Tensor]:
    assert lm.ndim == 3, lm.shape
    assert am.ndim == 3, am.shape
    assert lm.shape[0] == am.shape[0], (lm.shape, am.shape)
    assert lm.shape[2] == am.shape[2], (lm.shape, am.shape)
    (B, T, C) = am.shape
    S = lm.shape[1] - 1
    assert symbols.shape == (B, S), (symbols.shape, B, S)
    assert rnnt_type in ["regular", "modified", "constrained"], rnnt_type
    _validate_st_lengths(S, T, rnnt_type == "regular", boundary)

    am_max, _ = torch.max(am, dim=2, keepdim=True)
    lm_max, _ = torch.max(lm, dim=2, keepdim=True)
    am_probs = (am - am_max).exp()
    lm_probs = (lm - lm_max).exp()
    normalizers = (
        torch.matmul(lm_probs, am_probs.transpose(1, 2))
        + torch.finfo(lm_probs.dtype).tiny
    ).log()

    lmonly_normalizers = lm_probs.sum(dim=2, keepdim=True)
    unigram_lm = (
        torch.mean(lm_probs / lmonly_normalizers, dim=(0, 1), keepdim=True)
        + torch.finfo(lm_probs.dtype).tiny
    )
    amonly_normalizers = (
        torch.mv(am_probs.reshape(-1, C), unigram_lm.reshape(C))
        .reshape(B, T, 1)
        .log()
        + am_max
    )
    amonly_normalizers = amonly_normalizers.transpose(1, 2)
    unigram_lm = unigram_lm.log()
    lmonly_normalizers = lmonly_normalizers.log() + lm_max

    normalizers = normalizers + lm_max + am_max.transpose(1, 2)

    px_am = torch.gather(
        am.transpose(1, 2), dim=1, index=symbols.unsqueeze(2).expand(B, S, T)
    )

    if rnnt_type == "regular":
        px_am = torch.cat(
            (
                px_am,
                torch.full(
                    (B, S, 1),
                    float("-inf"),
                    device=px_am.device,
                    dtype=px_am.dtype,
                ),
            ),
            dim=2,
        )

    px_lm = torch.gather(lm[:, :S], dim=2, index=symbols.unsqueeze(-1))
    px_lm_unigram = torch.gather(
        unigram_lm.expand(B, S, C), dim=2, index=symbols.unsqueeze(-1)
    )

    px = px_am + px_lm
    px[:, :, :T] -= normalizers[:, :S, :]

    px_amonly = px_am + px_lm_unigram
    px_amonly[:, :, :T] -= amonly_normalizers
    px_lmonly = px_lm - lmonly_normalizers[:, :S, :]

    py_am = am[:, :, termination_symbol].unsqueeze(1)
    py_lm = lm[:, :, termination_symbol].unsqueeze(2)
    py = py_am + py_lm - normalizers

    py_lm_unigram = unigram_lm[0][0][termination_symbol]
    py_amonly = py_am + py_lm_unigram - amonly_normalizers
    py_lmonly = py_lm - lmonly_normalizers

    combined_scale = 1.0 - lm_only_scale - am_only_scale

    if lm_only_scale == 0.0:
        lm_only_scale = 1.0e-20
    if am_only_scale == 0.0:
        am_only_scale = 1.0e-20

    px_interp = (
        px * combined_scale
        + px_lmonly * lm_only_scale
        + px_amonly * am_only_scale
    )
    py_interp = (
        py * combined_scale
        + py_lmonly * lm_only_scale
        + py_amonly * am_only_scale
    )

    if rnnt_type == "regular":
        px_interp = fix_for_boundary(px_interp, boundary)
    elif rnnt_type == "constrained":
        px_interp = px_interp + py_interp[:, 1:, :]

    return (px_interp, py_interp)


def rnnt_loss_smoothed(
    lm: Tensor,
    am: Tensor,
    symbols: Tensor,
    termination_symbol: int,
    lm_only_scale: float = 0.1,
    am_only_scale: float = 0.1,
    boundary: Optional[Tensor] = None,
    rnnt_type: str = "regular",
    delay_penalty: float = 0.0,
    reduction: Optional[str] = "mean",
    return_grad: bool = False,
) -> Union[Tuple[Tensor, Tuple[Tensor, Tensor]], Tensor]:
    px, py = get_rnnt_logprobs_smoothed(
        lm=lm,
        am=am,
        symbols=symbols,
        termination_symbol=termination_symbol,
        lm_only_scale=lm_only_scale,
        am_only_scale=am_only_scale,
        boundary=boundary,
        rnnt_type=rnnt_type,
    )

    if delay_penalty > 0.0:
        B, S, T0 = px.shape
        T = T0 if rnnt_type != "regular" else T0 - 1
        if boundary is None:
            offset = torch.tensor(
                (T - 1) / 2, dtype=px.dtype, device=px.device
            ).expand(B, 1, 1)
        else:
            offset = (boundary[:, 3] - 1) / 2
        penalty = offset.reshape(B, 1, 1) - torch.arange(
            T0, device=px.device
        ).reshape(1, 1, T0)
        penalty = penalty * delay_penalty
        px = px + penalty.to(px.dtype)

    scores_and_grads = mutual_information_recursion(
        px=px, py=py, boundary=boundary, return_grad=return_grad
    )
    negated_loss = scores_and_grads[0] if return_grad else scores_and_grads
    if reduction == "none":
        loss = -negated_loss
    elif reduction == "mean":
        loss = -torch.mean(negated_loss)
    elif reduction == "sum":
        loss = -torch.sum(negated_loss)
    else:
        raise ValueError(
            f"reduction should be ('none' | 'mean' | 'sum'), given {reduction}"
        )
    return (loss, scores_and_grads[1]) if return_grad else loss
