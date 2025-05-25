import torchaudio
from src.logging.logger import logger

def is_valid_wav(file_path: str, min_duration: float = 5.0, max_duration: float = 10.0) -> bool:
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.shape[1] / float(sample_rate)
        logger.info(f"Audio duration: {duration:.2f}s, Sample rate: {sample_rate}")
        return min_duration <= duration <= max_duration
    except Exception as e:
        logger.error(f"Audio read error: {e}")
        return False