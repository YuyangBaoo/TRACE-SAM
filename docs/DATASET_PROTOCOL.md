# Dataset Protocol

TRACE-SAM-SR uses two data sources in the paper workflow:

- **Bridge Crack style labeled crack data** for topology-supervised SR
  fine-tuning, full TRACE-SAM joint training, evaluation, and SR augmentation.
- **Country Cement style HR concrete imagery** for image-only SR pretraining.

## Labeled Crack Layout

```text
data/bridge_crack/
  train/
    image/*.png
    label/*.png
  val/
    image/*.png
    label/*.png
  test/
    image/*.png
    label/*.png
```

Images and labels are paired by filename stem. The release loader supports
common image extensions (`png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`).

Set mask polarity in the YAML config:

```yaml
data:
  mask_foreground: dark   # dark crack pixels on light masks
  mask_threshold: 239
```

or:

```yaml
data:
  mask_foreground: light  # white/nonzero crack pixels
  mask_threshold: 127
```

## Image-only HR Layout

```text
data/country_cement/
  train/image/*.png
  val/image/*.png
```

The loader also accepts flat directories. This source does not require masks;
zero topology maps are returned so the same SR training loop can be reused.

## Leakage Rule

Ground-truth masks and topology maps may be used only for training losses,
data-pair construction, or explicitly marked upper-bound analysis. Normal
inference and evaluation use LR images, LR-up images, learned TRACE-SAM-SR
outputs, and SAM-compatible prompts only.
