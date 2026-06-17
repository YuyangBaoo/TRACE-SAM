"""Command-line wrapper for TRACE-SAM-SR training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_sam.scripts.train_trace_sam_sr import main


if __name__ == "__main__":
    main()
