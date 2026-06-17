# TRACE-SAM-SR Model Card

## Model Details

TRACE-SAM-SR is a fracture-field guided super-resolution and crack segmentation
framework for low-resolution concrete inspection imagery. The release code
contains the TRACE-SAM-SR model, SAM-based recognition branch, synthetic demo
workflow, and paper-default configuration.

The full TRACE-SAM recognition branch requires a SAM ViT-B checkpoint at runtime.
SAM weights are not redistributed in this repository.

## Intended Use

TRACE-SAM-SR is intended for research on concrete crack restoration,
segmentation, data augmentation, and morphology-aware evaluation under
low-resolution inspection conditions. The code is suitable for reproducing
training/evaluation protocols and for prototyping on labeled crack imagery.

## Out-of-Scope Use

The model should not be used as the sole basis for safety-critical structural
assessment, maintenance prioritization, or public-risk decisions without
qualified human review, target-domain calibration, and independent validation.

## Training and Evaluation Context

The accompanying manuscript evaluates TRACE-SAM-SR under test-time restoration
and training-time SR augmentation pathways. Reported metrics include PSNR, SSIM,
FID, Dice F1, Boundary F1, clDice, crack length error, width error, and
background hallucination indicators.

The synthetic `demo_data/` workflow included in this repository is only for
software smoke testing and is not used for paper performance claims.

## Limitations

- Performance depends on the target inspection protocol, mask polarity, camera
  resolution, and low-resolution degradation process.
- The method can fail when LR evidence no longer contains recoverable crack
  information.
- The SAM-based recognition branch requires the user to provide compatible SAM
  weights and may need calibration for new crack datasets.
- The release does not include the full training datasets or large checkpoints.

## Responsible Release Notes

Users should report the exact config, checkpoint hashes, dataset split, mask
polarity, degradation ID, PyTorch/CUDA versions, GPU model, random seed, and
command line when publishing results.

