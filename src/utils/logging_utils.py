import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_run_logger(run_name: str, *, output_dir: str = "logs") -> logging.Logger:
    """
    Create a file logger for a single run. Safe to call multiple times.
    """
    name = str(run_name).strip().replace(" ", "_") or "run"
    logger = logging.getLogger(f"hfo.{name}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"{name}_{ts}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("Log file: %s", str(log_path))
    return logger


def log_section(logger: logging.Logger, title: Optional[str] = None, *, width: int = 72) -> None:
    line = "=" * int(width)
    if title:
        logger.info(line)
        logger.info("%s", str(title))
    logger.info(line)
