import torch
from nemo.collections.asr.models import EncDecCTCModel
from src.logging.logger import logger

def convert_nemo_to_onnx(nemo_model_path: str, export_path: str = "model/nemo_conformer_exported.onnx"):
    logger.info(f"Restoring model from {nemo_model_path}")
    asr_model = EncDecCTCModel.restore_from(nemo_model_path).eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    logger.info("Exporting model to ONNX format...")
    asr_model.export(output=export_path, onnx_opset_version=13)
    logger.info(f"Model successfully exported to {export_path}")
