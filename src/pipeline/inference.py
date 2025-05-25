import onnxruntime as ort
import numpy as np
import torch
from nemo.collections.asr.models import EncDecCTCModel
from ...app.utils import preprocess_audio

# ONNX runtime session
onnx_model_path = "model/nemo_conformer_optimized_ORT_ENABLE_ALL_T4.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# NeMo model for decoding (lightweight, only used for decoding)
decoder_model = EncDecCTCModel.restore_from("model/stt_hi_conformer_ctc_medium.nemo")
decoder_model.eval()

def infer_transcription(audio_path):
    audio_signal, length = preprocess_audio(audio_path)
    inputs = {"audio_signal": audio_signal, "length": length}

    preds = session.run(None, inputs)
    log_probs = preds[0][0]
    pred_ids = np.argmax(log_probs, axis=-1)

    pred_tensor = torch.tensor(pred_ids).unsqueeze(0)
    text = decoder_model.decoding.ctc_decoder_predictions_tensor(pred_tensor)[0]
    return text
