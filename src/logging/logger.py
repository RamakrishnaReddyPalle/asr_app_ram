import logging

logger = logging.getLogger("asr_app")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(ch)
