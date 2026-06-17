# AGENTS.md - TRACE-SAM-SR Instructions

This repository is the cleaned TRACE-SAM-SR workflow for concrete crack
super-resolution and segmentation.

## Rules

- Keep generated outputs inside `runs/`, `results/`, or `playground/results/`.
- Keep private datasets outside git; use `data/` or `playground/inputs/`.
- Use `configs/demo_cpu.yaml` for smoke tests.
- Use `configs/paper_trace_sam_sr.yaml` for paper-style runs.
- Use `run_trace_sam.ps1` on Windows and `run_trace_sam.sh` on Linux as the
  maintained workflow entry points.
- Do not add duplicate one-off run scripts in the repository root.
- Do not pass ground-truth masks or topology maps to inference/evaluation.
- Evaluation must use the held-out crack test split.
- Augmentation patches must be generated only from training images.

## Validation

```powershell
powershell -ExecutionPolicy Bypass -File .\run_trace_sam.ps1 -Preset dry -Device cpu
```

```bash
./run_trace_sam.sh --preset dry --device cpu
```

```bash
python -m compileall trace_sam tools
```
