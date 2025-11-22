from contextlib import contextmanager
import logging
from tqdm import tqdm
import sys


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)

@contextmanager
def tqdm_logging_redirect():
    """Context manager that redirect logging to tqdm.write"""
    handler = TqdmLoggingHandler()
    root_logger = logging.getLogger()

    old_handlers = root_logger.handlers[:]

    for old_handler in old_handlers:
        root_logger.removeHandler(old_handler)
    root_logger.addHandler(handler)

    try: 
        yield
    finally:
        root_logger.removeHandler(handler)
        for old_handler in old_handlers:
            root_logger.addHandler(old_handler)
