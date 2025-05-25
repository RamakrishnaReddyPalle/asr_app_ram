import os
import time
import torch
import onnxruntime as ort
import numpy as np
import pandas as pd
from nemo.collections.asr.models import EncDecCTCModel
from src.utils.audio_pre_processing import preprocess_audio_for_onnx
from src.logging.logger import logger

def benchmark_and_save_best_model(audio_path: str, onnx_export_path: str):
    audio_signal, length, waveform = preprocess_audio_for_onnx(audio_path)
    audio_signal = audio_signal.astype(np.float32)
    length = length.astype(np.int64)

    onnx_input = {
        "audio_signal": audio_signal,
        "length": length
    }

    search_space = [
        {"opt_level": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "intra_threads": 1},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "intra_threads": 1},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "intra_threads": 1},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "intra_threads": 1},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "intra_threads": 2},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "intra_threads": 4},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "intra_threads": 8},
        {"opt_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "intra_threads": 16},
    ]

    results = []
    for cfg in search_space:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = cfg["opt_level"]
        sess_options.intra_op_num_threads = cfg["intra_threads"]

        session = ort.InferenceSession(onnx_export_path, sess_options, providers=["CPUExecutionProvider"])
        _ = session.run(None, onnx_input)

        start = time.time()
        for _ in range(5):
            preds = session.run(None, onnx_input)
        end = time.time()

        avg_time = (end - start) / 5
        log_probs = preds[0][0]
        pred_ids = np.argmax(log_probs, axis=-1)

        asr_model = EncDecCTCModel.restore_from("model/stt_hi_conformer_ctc_medium.nemo").eval()
        text = asr_model.decoding.ctc_decoder_predictions_tensor(torch.tensor(pred_ids).unsqueeze(0))[0]

        results.append({
            "config": f"{cfg['opt_level'].name}-T{cfg['intra_threads']}",
            "inference_time": avg_time,
            "prediction": text
        })

    df = pd.DataFrame(results)
    best = df[df["config"] != "NeMo_PyTorch"].sort_values("inference_time").iloc[0]
    best_name = best["config"].replace("-", "_")
    best_path = f"model/nemo_conformer_optimized_{best_name}.onnx"
    
    os.rename(onnx_export_path, best_path)
    logger.info(f"Best config: {best_name}, model saved to {best_path}")

    return best_path, best["prediction"]

def run_inference_with_model(audio_path, onnx_model_path):
    # Preprocessing audio to match ONNX model input requirements
    input_tensor = preprocess_audio_for_onnx(audio_path)  # shape: (1, features, time)

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor.numpy()})
    prediction = np.argmax(output[0], axis=-1)

    # TODO: Converting prediction to actual transcript using a decoder or label map
    return prediction.tolist()