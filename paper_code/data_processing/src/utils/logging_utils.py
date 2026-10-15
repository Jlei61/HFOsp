"""Quiet standard logging for the detector."""
import logging

def get_run_logger(run_name, **kwargs):
    logger = logging.getLogger('hfo.' + str(run_name))
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger

def log_section(logger, title=None, **kwargs):
    if title:
        logger.info(title)
