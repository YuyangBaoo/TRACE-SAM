# Third-party Notices

TRACE-SAM-SR vendors the model-building components of Segment Anything under
`trace_sam/vendors/segment_anything`. Those files are derived from Meta AI's
Segment Anything implementation and retain the original copyright headers.

Segment Anything is licensed under Apache License 2.0. A copy is provided under
`LICENSES/Apache-2.0.txt`.

No pretrained SAM weights are included in this package. Download the SAM ViT-B
checkpoint from Meta's official release and set `paths.sam_checkpoint` in the
YAML config before training or evaluating the full recognition branch.

