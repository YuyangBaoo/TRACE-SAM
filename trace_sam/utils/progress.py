"""Small terminal progress helpers backed by tqdm."""
from __future__ import annotations

import sys
from typing import Any

from tqdm.auto import tqdm


def format_seconds(seconds: float | int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def stage_banner(index: int, total: int, name: str) -> None:
    print(f"\n{'=' * 18} [{index}/{total}] {name} {'=' * 18}", flush=True)


class ProgressBar:
    def __init__(
        self,
        total: int,
        desc: str,
        unit: str = "it",
        width: int = 28,
        min_interval: float = 1.0,
    ) -> None:
        self.total = max(1, int(total))
        self.desc = str(desc)
        self.unit = str(unit)
        self.count = 0
        self.closed = False
        self.bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            dynamic_ncols=True,
            mininterval=float(min_interval),
            ascii=True,
            leave=True,
            file=sys.stdout,
        )

    def update(self, n: int = 1, **metrics: Any) -> None:
        step = max(0, min(int(n), self.total - self.count))
        self.count += step
        if metrics:
            formatted = {}
            for key, value in metrics.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.6g}"
                else:
                    formatted[key] = value
            self.bar.set_postfix(formatted, refresh=False)
        if step:
            self.bar.update(step)

    def close(self) -> None:
        if not self.closed:
            self.bar.close()
            self.closed = True
