"""Project-wide pytest setup that avoids Windows temporary-folder lock races."""

from __future__ import annotations

import os
import time
from pathlib import Path


def pytest_configure(config) -> None:
    """Give every run a new temp root so pytest never removes a watched folder."""

    if config.option.basetemp is None:
        temp_parent = Path(__file__).parent / "tests" / ".pytest-runs"
        temp_parent.mkdir(parents=True, exist_ok=True)
        run_id = f"pytest-run-{os.getpid()}-{time.time_ns()}"
        config.option.basetemp = str(temp_parent / run_id)
