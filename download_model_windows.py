import os
import urllib.request

model_dir = "model"
model_file = "stt_hi_conformer_ctc_medium.nemo"
url = f"https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/stt_hi_conformer_ctc_medium/1.6.0/files?redirect=true&path={model_file}"

os.makedirs(model_dir, exist_ok=True)
output_path = os.path.join(model_dir, model_file)

print(f"Downloading model to {output_path}...")

try:
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
except Exception as e:
    print("Failed to download model:", e)
