import torch
import torchaudio
import numpy as np
from src.logging.logger import logger

def preprocess_audio_for_onnx(file_path, target_sr=16000, n_mels=80):
    logger.info(f"Loading and preprocessing audio from: {file_path}")
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        logger.debug("Converted stereo to mono")

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        logger.debug(f"Resampled audio from {sample_rate}Hz to {target_sr}Hz")

    waveform = waveform / waveform.abs().max()
    logger.debug("Normalized waveform")

    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_mels=n_mels,
        n_fft=400,
        win_length=400,
        hop_length=160
    )
    mel_spec = mel_spec_transform(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-9)

    audio_signal = log_mel_spec.numpy()
    length = np.array([audio_signal.shape[2]], dtype=np.int64)

    logger.info(f"Preprocessing complete. Audio shape: {audio_signal.shape}")
    return audio_signal, length, waveform
