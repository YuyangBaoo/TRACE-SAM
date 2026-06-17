# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the original
# Segment Anything repository. Only the SAM model-building components required
# by TRACE-SAM are vendored here.

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)

sam_model_checkpoint = {
    "default": "sam_vit_h_4b8939.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
