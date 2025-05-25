# Project Description

## ✅ Features Successfully Implemented

1. **Hindi ASR Pipeline using NVIDIA NeMo**
   I successfully integrated the `stt_hi_conformer_ctc_medium` model from NVIDIA NeMo—a pretrained CTC-based conformer model capable of transcribing Hindi speech into text. It achieves high accuracy in transcriptions for general-domain Hindi audio.

2. **ONNX Model Export and Optimization**
   Instead of using `torch.onnx.export` (which failed due to NeMo’s reliance on `torch.stft`, a function not yet fully supported in ONNX export), I directly exported the `.nemo` model to ONNX using NeMo’s built-in export API. This ensured compatibility and completeness of the model graph.

3. **Creative Optimization Search Space Definition**
   A major highlight of the project was the **search space exploration for ONNX graph optimizations**.
   I defined a set of optimization strategies (e.g., `basic`, `extended`, `all`, and ORT-specific configurations with different threading levels and execution modes).
   Each configuration was applied to the base ONNX model, and its inference time was benchmarked using a standardized test input.
   The results were plotted for comparison, and the **best-performing optimized ONNX model** was selected, saved as `stt_hi_conformer_ctc_medium_best.onnx`, and used for deployment.

4. **FastAPI Deployment**
   A clean FastAPI application was created, exposing an endpoint `/transcribe` that accepts a `.wav` file and returns its transcription. The Swagger UI (`/docs`) provides a user-friendly way to test and understand the API.

5. **Robust Preprocessing and Audio Validation**
   To ensure valid inputs, I added automatic:

   * Audio format checking,
   * Duration verification (5–10 seconds),
   * Resampling and channel adjustments, if needed.

6. **Dockerized Deployment**
   The full stack is containerized using a `python:3.11-slim` base image. All necessary dependencies (e.g., `ffmpeg`, `libsndfile`, NeMo, ONNX Runtime) are included. Users can run the API on any system supporting Docker with just two commands.

7. **Helper Scripts for Model Setup**
   A `download_model.sh` (for macOS/Linux) and `download_model.bat` (for Windows) script is provided to fetch the `.nemo` model and place it into the `model/` directory.

8. **Postman and cURL API Testing**
   I included usage examples with `cURL` and Postman to show how the `/transcribe` endpoint works. Local screenshots (placed in `assets/`) showcase this.

9. **Inference Time Visualization**
   A plot visualizing the performance of various optimization levels and strategies was generated and embedded in the `README` to help users understand the trade-offs.

---

## ⚠️ Issues Encountered

* **ONNX Export via PyTorch**
  I initially attempted to convert the NeMo model to ONNX using `torch.onnx.export`. However, this failed due to the model’s use of `torch.stft`, which is not fully ONNX-compatible.
  As a workaround, I used NeMo’s own `Exportable` interface to directly export the `.nemo` model to ONNX.

* **ONNX Optimization Trade-offs**
  Some optimization levels reduced inference time but slightly compromised model output quality (e.g., logits flattening errors). To mitigate this, I empirically tested each variant for both **accuracy** and **inference latency** before selecting the best version.

---

## 🚫 Features Not Implemented & Why

* **Streaming ASR (Real-time)**
  I didn't implement streaming inference (via WebSockets or live audio chunks) due to time constraints. It also required additional architectural changes to the FastAPI server.

* **Frontend Interface**
  A GUI or web interface (e.g., Gradio or React frontend) was not prioritized, as the backend functionality was the primary focus.

---

## Creative Solutions and Insights

* **Optimization Strategy as a Search Problem**
  Rather than blindly applying a single ONNX optimization pass, I treated the optimization process like a **hyperparameter search space**.
  I defined several configurations varying:

  * ONNX optimization levels (`basic`, `extended`, `all`)
  * Threading (`intra_op_num_threads`, `inter_op_num_threads`)
  * Execution modes (`sequential`, `parallel`)
  * Graph-level settings (shape inference, constant folding)

  I benchmarked each configuration on the same test input to collect latency metrics and identify the fastest one.
  This approach added **experimental rigor** and made the deployment both fast and robust.

---

## Overcoming the Challenges

* For the `torch.onnx.export` issue, I learned to rely on framework-specific export tools—in this case, NeMo’s `.nemo → ONNX` pathway.
* For audio preprocessing inconsistencies across OSes, I added an abstracted `utils.py` module that automatically handles format, sample rate, and mono-channel conversion.
* For future scalability (streaming or real-time inference), I plan to integrate FastAPI’s WebSocket support and switch to a streaming-capable ASR model.

---

## Known Limitations and Assumptions

* The model is not domain-finetuned. In niche audio contexts (e.g., medical or legal Hindi speech), performance may degrade.
* Inputs must be WAV files of **5–10 seconds**. Other formats or durations are automatically rejected.
* The `/transcribe` endpoint supports only single-file inference as of now.
* The Docker image assumes an x86\_64-based system; compatibility with ARM-based machines (e.g., Apple Silicon) is unverified.
* The user must manually run the `download_model.sh` or `.bat` script before running the app to ensure the model file is present.


