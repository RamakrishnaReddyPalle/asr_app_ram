from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import numpy as np
import torch
import onnxruntime as ort

from src.utils.audio_pre_processing import preprocess_audio_for_onnx
from src.utils.save_temp_wav import convert_to_wav_and_resample
from src.utils.is_valid_wav import is_valid_wav
from src.logging.logger import logger
from nemo.collections.asr.models import EncDecCTCModel

# Constants
MODEL_PATH = "model/best_optimized_onnx_model.onnx"
NEMO_MODEL_PATH = "model/stt_hi_conformer_ctc_medium.nemo" 

app = FastAPI(
    title="Hindi ASR App",
    description=(
        "Upload a short audio clip (5–10 seconds).\n"
        "- All formats supported (e.g., mp3, m4a, wav, ogg).\n"
        "- Audio will be automatically converted to mono WAV at 16kHz.\n"
        "- Ideal for transcribing Hindi keywords."
    )
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Hindi ASR system!",
        "usage": "Use POST /transcribe with an audio file (mp3, wav, m4a, etc.). Length must be 5 to 10 seconds."
    }

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")

        # Conversion and validation
        temp_wav_path = convert_to_wav_and_resample(file)
        if not is_valid_wav(temp_wav_path):
            return JSONResponse(status_code=400, content={"error": "Audio must be between 5 and 10 seconds."})

        # Audio Processing for ONNX
        audio_signal, length, waveform = preprocess_audio_for_onnx(temp_wav_path)
        audio_signal = audio_signal.astype(np.float32)
        length = length.astype(np.int64)

        onnx_input = {
            "audio_signal": audio_signal,
            "length": length
        }

        # Load and run ONNX
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        preds = session.run(None, onnx_input)

        log_probs = preds[0][0]
        pred_ids = np.argmax(log_probs, axis=-1)

        # NeMo's decoder
        asr_model = EncDecCTCModel.restore_from(NEMO_MODEL_PATH).eval()
        text = asr_model.decoding.ctc_decoder_predictions_tensor(torch.tensor(pred_ids).unsqueeze(0))[0]

        return {"prediction": text, "onnx_model_used": os.path.basename(MODEL_PATH)}

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

