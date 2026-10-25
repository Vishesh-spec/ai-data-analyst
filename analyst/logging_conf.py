import logging, os
from logging.handlers import RotatingFileHandler


def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); root.addHandler(sh)
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=2)
    fh.setFormatter(fmt); root.addHandler(fh)
