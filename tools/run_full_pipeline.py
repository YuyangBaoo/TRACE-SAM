"""Command-line wrapper for the one-click TRACE-SAM workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_sam.scripts.run_full_pipeline import main


if __name__ == "__main__":
    main()
