# Ablation Results Summary

This table reports the completed TRACE-SAM-SR ablation outputs used for the release evidence package. Empty cells mean the metric was not produced by that branch.

| group | variant | psnr | ssim | lpips | fid | dice_f1 | precision | recall | boundary_f1 | cldice | length_error | fracture_field_mae | background_hallucination_index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sr_component | no_fracture_field | 24.7817 | 0.8056 |  |  | 0.0000 |  |  | 0.0000 | 0.0000 | 1.0000 | 0.0582 | 0.0003 |
| sr_component | no_gated_refiner | 24.8424 | 0.8092 |  |  | 0.3073 |  |  | 0.5565 | 0.2537 | 0.4004 | 0.0362 | 0.0066 |
| sr_component | no_structure_losses | 24.8259 | 0.8124 |  |  | 0.0559 |  |  | 0.2469 | 0.0476 | 0.8581 | 0.0646 | 0.0094 |
| recognition_component | no_sr_uncertainty |  |  |  |  | 0.7541 | 0.7523 | 0.7854 | 0.7638 | 0.6014 | 0.1573 |  |  |
| recognition_component | no_thin_line_refiner |  |  |  |  | 0.7539 | 0.7479 | 0.7894 | 0.7609 | 0.5983 | 0.1590 |  |  |
| augmentation_control | bicubic_lr_up_control |  |  |  |  | 0.7533 | 0.7553 | 0.7812 | 0.7639 | 0.6063 | 0.1570 |  |  |
| reference | trace_sam_sr_full |  |  |  |  | 0.7542 | 0.7505 | 0.7878 | 0.7621 | 0.6021 | 0.1572 |  |  |
