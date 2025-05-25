import os
import uuid
import torchaudio
import torchaudio.transforms as T
from src.logging.logger import logger

def convert_to_wav_and_resample(uploaded_file, output_dir="test", sample_rate=16000) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # To generate unique filename
    temp_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.wav")

    # To save uploaded file temporarily
    temp_original_path = os.path.join(output_dir, f"{uuid.uuid4().hex}_orig")
    with open(temp_original_path, "wb") as f:
        f.write(uploaded_file.file.read())

    # torchaudio (auto detects format)
    waveform, original_sr = torchaudio.load(temp_original_path)
    logger.info(f"Loaded file with shape {waveform.shape} and sample rate {original_sr}")

    # stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if original_sr != sample_rate:
        resampler = T.Resample(orig_freq=original_sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Save as WAV
    torchaudio.save(temp_path, waveform, sample_rate)
    logger.info(f"Saved converted WAV to {temp_path}")

    os.remove(temp_original_path)

    return temp_path
