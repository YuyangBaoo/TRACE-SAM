"""Command-line wrapper for TRACE-SAM-SR SR-image evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_sam.scripts.evaluate_trace_sr_predictions import main


if __name__ == "__main__":
    main()
