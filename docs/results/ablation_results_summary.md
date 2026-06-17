# Ablation results summary

This table is generated from completed ablation outputs and copied baseline metrics. Empty cells mean the metric was not produced by that branch.

| group | variant | psnr | ssim | lpips | fid | dice_f1 | precision | recall | boundary_f1 | cldice | length_error | fracture_field_mae | background_hallucination_index | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sr_ablation | no_fracture_field | 24.7817 | 0.8056 |  |  | 0.0000 |  |  | 0.0000 | 0.0000 | 1.0000 | 0.0582 | 0.0003 | results/crackguard_diffsr/sr_ablation/no_fracture_field/seed_1234/inference/test/D0/metrics |
| sr_ablation | no_gated_refiner | 24.8424 | 0.8092 |  |  | 0.3073 |  |  | 0.5565 | 0.2537 | 0.4004 | 0.0362 | 0.0066 | results/crackguard_diffsr/sr_ablation/no_gated_refiner/seed_1234/inference/test/D0/metrics |
| sr_ablation | no_structure_losses | 24.8259 | 0.8124 |  |  | 0.0559 |  |  | 0.2469 | 0.0476 | 0.8581 | 0.0646 | 0.0094 | results/crackguard_diffsr/sr_ablation/no_structure_losses/seed_1234/inference/test/D0/metrics |
| recognition_ablation | no_sr_uncertainty |  |  |  |  | 0.7541 | 0.7523 | 0.7854 | 0.7638 | 0.6014 | 0.1573 |  |  | results/crackguard_diffsr/recognition_ablation/no_sr_uncertainty/seed_1234/eval/offline_augmentation/test/D0/metrics_summary.json |
| recognition_ablation | no_thin_line_refiner |  |  |  |  | 0.7539 | 0.7479 | 0.7894 | 0.7609 | 0.5983 | 0.1590 |  |  | results/crackguard_diffsr/recognition_ablation/no_thin_line_refiner/seed_1234/eval/offline_augmentation/test/D0/metrics_summary.json |
| augmentation_control | bicubic_aug |  |  |  |  | 0.7533 | 0.7553 | 0.7812 | 0.7639 | 0.6063 | 0.1570 |  |  | results/crackguard_diffsr/augmentation_control/bicubic_aug/seed_1234/eval/offline_augmentation/test/D0/metrics_summary.json |
| baseline | ours_v3 |  |  |  |  | 0.7542 | 0.7505 | 0.7878 | 0.7621 | 0.6021 | 0.1572 |  |  | playground/results/trace_sam_sr/full_image_aug_v3_unfreeze_0611/eval/offline_augmentation/test/D0/metrics_summary.json |
