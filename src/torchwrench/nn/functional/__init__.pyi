from torch.nn.functional import *

from .activation import (
    log_softmax_multidim as log_softmax_multidim,
)
from .activation import (
    softmax_multidim as softmax_multidim,
)
from .checksum import checksum_any as checksum_any
from .cropping import crop_dim as crop_dim
from .cropping import crop_dims as crop_dims
from .indices import (
    get_inverse_perm as get_inverse_perm,
)
from .indices import (
    get_perm_indices as get_perm_indices,
)
from .indices import (
    insert_at_indices as insert_at_indices,
)
from .indices import (
    randperm_diff as randperm_diff,
)
from .indices import (
    remove_at_indices as remove_at_indices,
)
from .make import (
    as_device as as_device,
)
from .make import (
    as_dtype as as_dtype,
)
from .make import (
    as_generator as as_generator,
)
from .make import (
    get_default_device as get_default_device,
)
from .make import (
    get_default_dtype as get_default_dtype,
)
from .make import (
    get_default_generator as get_default_generator,
)
from .make import (
    set_default_dtype as set_default_dtype,
)
from .make import (
    set_default_generator as set_default_generator,
)
from .mask import (
    generate_square_subsequent_mask as generate_square_subsequent_mask,
)
from .mask import (
    lengths_to_non_pad_mask as lengths_to_non_pad_mask,
)
from .mask import (
    lengths_to_pad_mask as lengths_to_pad_mask,
)
from .mask import (
    lengths_to_ratios as lengths_to_ratios,
)
from .mask import (
    masked_equal as masked_equal,
)
from .mask import (
    masked_mean as masked_mean,
)
from .mask import (
    masked_sum as masked_sum,
)
from .mask import (
    non_pad_mask_to_lengths as non_pad_mask_to_lengths,
)
from .mask import (
    non_pad_mask_to_ratios as non_pad_mask_to_ratios,
)
from .mask import (
    pad_mask_to_lengths as pad_mask_to_lengths,
)
from .mask import (
    pad_mask_to_ratios as pad_mask_to_ratios,
)
from .mask import (
    ratios_to_lengths as ratios_to_lengths,
)
from .mask import (
    ratios_to_non_pad_mask as ratios_to_non_pad_mask,
)
from .mask import (
    ratios_to_pad_mask as ratios_to_pad_mask,
)
from .mask import (
    tensor_to_lengths as tensor_to_lengths,
)
from .mask import (
    tensor_to_non_pad_mask as tensor_to_non_pad_mask,
)
from .mask import (
    tensor_to_pad_mask as tensor_to_pad_mask,
)
from .mask import (
    tensor_to_tensors_list as tensor_to_tensors_list,
)
from .mask import (
    tensors_list_to_lengths as tensors_list_to_lengths,
)
from .multiclass import (
    index_to_name as index_to_name,
)
from .multiclass import (
    index_to_onehot as index_to_onehot,
)
from .multiclass import (
    name_to_index as name_to_index,
)
from .multiclass import (
    name_to_onehot as name_to_onehot,
)
from .multiclass import (
    one_hot as one_hot,
)
from .multiclass import (
    onehot_to_index as onehot_to_index,
)
from .multiclass import (
    onehot_to_name as onehot_to_name,
)
from .multiclass import (
    probs_to_index as probs_to_index,
)
from .multiclass import (
    probs_to_name as probs_to_name,
)
from .multiclass import (
    probs_to_onehot as probs_to_onehot,
)
from .multilabel import (
    indices_to_multihot as indices_to_multihot,
)
from .multilabel import (
    indices_to_multinames as indices_to_multinames,
)
from .multilabel import (
    multi_indices_to_multihot as multi_indices_to_multihot,
)
from .multilabel import (
    multi_indices_to_multinames as multi_indices_to_multinames,
)
from .multilabel import (
    multihot_to_indices as multihot_to_indices,
)
from .multilabel import (
    multihot_to_multi_indices as multihot_to_multi_indices,
)
from .multilabel import (
    multihot_to_multinames as multihot_to_multinames,
)
from .multilabel import (
    multinames_to_indices as multinames_to_indices,
)
from .multilabel import (
    multinames_to_multi_indices as multinames_to_multi_indices,
)
from .multilabel import (
    multinames_to_multihot as multinames_to_multihot,
)
from .multilabel import (
    probs_to_indices as probs_to_indices,
)
from .multilabel import (
    probs_to_multi_indices as probs_to_multi_indices,
)
from .multilabel import (
    probs_to_multihot as probs_to_multihot,
)
from .multilabel import (
    probs_to_multinames as probs_to_multinames,
)
from .new import (
    arange as arange,
)
from .new import (
    empty as empty,
)
from .new import (
    full as full,
)
from .new import (
    ones as ones,
)
from .new import (
    rand as rand,
)
from .new import (
    randint as randint,
)
from .new import (
    randperm as randperm,
)
from .new import (
    zeros as zeros,
)
from .others import (
    average_power as average_power,
)
from .others import (
    cat as cat,
)
from .others import (
    concat as concat,
)
from .others import (
    count_parameters as count_parameters,
)
from .others import (
    deep_equal as deep_equal,
)
from .others import (
    find as find,
)
from .others import (
    get_ndim as get_ndim,
)
from .others import (
    get_shape as get_shape,
)
from .others import (
    mse as mse,
)
from .others import (
    ndim as ndim,
)
from .others import (
    nelement as nelement,
)
from .others import (
    prod as prod,
)
from .others import (
    ranks as ranks,
)
from .others import (
    rmse as rmse,
)
from .others import (
    shape as shape,
)
from .others import (
    stack as stack,
)
from .padding import (
    cat_padded_batch as cat_padded_batch,
)
from .padding import (
    pad_and_stack_rec as pad_and_stack_rec,
)
from .padding import (
    pad_dim as pad_dim,
)
from .padding import (
    pad_dims as pad_dims,
)
from .powerset import (
    multilabel_to_powerset as multilabel_to_powerset,
)
from .powerset import (
    powerset_to_multilabel as powerset_to_multilabel,
)
from .predicate import (
    all_eq as all_eq,
)
from .predicate import (
    all_ne as all_ne,
)
from .predicate import (
    is_complex as is_complex,
)
from .predicate import (
    is_convertible_to_tensor as is_convertible_to_tensor,
)
from .predicate import (
    is_floating_point as is_floating_point,
)
from .predicate import (
    is_full as is_full,
)
from .predicate import (
    is_sorted as is_sorted,
)
from .predicate import (
    is_stackable as is_stackable,
)
from .predicate import (
    is_unique as is_unique,
)
from .segments import (
    activity_to_segments as activity_to_segments,
)
from .segments import (
    activity_to_segments_list as activity_to_segments_list,
)
from .segments import (
    segments_list_to_activity as segments_list_to_activity,
)
from .segments import (
    segments_to_activity as segments_to_activity,
)
from .segments import (
    segments_to_segments_list as segments_to_segments_list,
)
from .transform import (
    as_tensor as as_tensor,
)
from .transform import (
    flatten as flatten,
)
from .transform import (
    identity as identity,
)
from .transform import (
    move_to as move_to,
)
from .transform import (
    move_to_rec as move_to_rec,
)
from .transform import (
    pad_and_crop_dim as pad_and_crop_dim,
)
from .transform import (
    recursive_to as recursive_to,
)
from .transform import (
    repeat_interleave_nd as repeat_interleave_nd,
)
from .transform import (
    resample_nearest_freqs as resample_nearest_freqs,
)
from .transform import (
    resample_nearest_rates as resample_nearest_rates,
)
from .transform import (
    resample_nearest_steps as resample_nearest_steps,
)
from .transform import (
    shuffled as shuffled,
)
from .transform import (
    squeeze as squeeze,
)
from .transform import (
    squeeze_ as squeeze_,
)
from .transform import (
    squeeze_copy as squeeze_copy,
)
from .transform import (
    to_item as to_item,
)
from .transform import (
    to_tensor as to_tensor,
)
from .transform import (
    top_k as top_k,
)
from .transform import (
    top_p as top_p,
)
from .transform import (
    topk as topk,
)
from .transform import (
    transform_drop as transform_drop,
)
from .transform import (
    unsqueeze as unsqueeze,
)
from .transform import (
    unsqueeze_ as unsqueeze_,
)
from .transform import (
    unsqueeze_copy as unsqueeze_copy,
)
from .transform import (
    view_as_complex as view_as_complex,
)
from .transform import (
    view_as_real as view_as_real,
)
