# isort:skip_file

import os as _os

# ``k2`` only ships CPU / CUDA kernels -- there is no Ascend NPU backend.  When
# k2 cannot be imported (the usual situation on NPU), or when explicitly
# requested, install a minimal torch-based replacement for the part of the k2
# API used by the training / fine-tuning path.  Set ``ICEFALL_USE_TORCH_K2=1``
# to force it even when a real k2 installation is present.
_use_torch_k2 = _os.environ.get("ICEFALL_USE_TORCH_K2", "0") == "1"
if not _use_torch_k2:
    try:  # use the real k2 package when it is available (CPU / CUDA builds)
        import k2 as _k2  # noqa: F401
    except ImportError:
        _use_torch_k2 = True

if _use_torch_k2:
    from . import k2_compat as _k2_compat

    _k2_compat.install()

from . import checkpoint, decode, dist, env, utils

from .byte_utils import (
    byte_decode,
    byte_encode,
    smart_byte_decode,
)

from .checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
    remove_checkpoints,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
)

from .context_graph import ContextGraph, ContextState

from .decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)

from .dist import (
    cleanup_dist,
    setup_dist,
)

from .env import (
    get_env_info,
    get_git_branch_name,
    get_git_date,
    get_git_sha1,
)

from .utils import (
    AttributeDict,
    MetricsTracker,
    add_eos,
    add_sos,
    concat,
    encode_supervisions,
    get_alignments,
    get_executor,
    get_texts,
    is_cjk,
    is_jit_tracing,
    is_module_available,
    l1_norm,
    l2_norm,
    linf_norm,
    load_alignments,
    make_pad_mask,
    measure_gradient_norms,
    measure_weight_norms,
    optim_step_and_measure_param_change,
    save_alignments,
    setup_logger,
    store_transcripts,
    str2bool,
    subsequent_chunk_mask,
    tokenize_by_CJK_char,
    tokenize_by_ja_char,
    write_error_stats,
)

from .ngram_lm import NgramLm, NgramLmStateCost

from .lm_wrapper import LmScorer
