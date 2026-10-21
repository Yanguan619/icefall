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

"""Thin re-export of :mod:`icefall.npu_compat`.

The device / AMP helpers live inside the ``icefall`` package so that they can
also be used by ``icefall.dist``.  The recipe scripts import them from this
sibling module for convenience.
"""

from icefall.npu_compat import (  # noqa: F401
    GradScaler,
    autocast,
    custom_bwd,
    custom_fwd,
    device_count,
    device_type,
    dist_backend,
    get_autocast_dtype,
    get_device,
    get_rng_state,
    has_accelerator,
    is_autocast_enabled,
    is_bf16_supported,
    is_cuda_available,
    is_npu_available,
    max_memory_allocated,
    set_device,
)
