# TRACE-SAM Playground

`playground/` is reserved for local, non-versioned inputs and outputs.

Recommended local layout:

```text
playground/
  inputs/
    bridge_crack/          # train/val/test image-label data
    country_cement/        # image-only HR data for SR pretraining
    checkpoints/           # optional local checkpoint mirror
  results/
    trace_sam_sr/          # full workflow outputs
    crackguard_diffsr/     # ablation outputs
```

The release configs use `data/`, `checkpoints/`, and `runs/` by default so that
new users can start without creating this folder. Use `playground/` when you
want to keep large private data outside the publishable tree.

