import torch
from onnx2torch import convert
import onnx
from pathlib import Path
import os
import tempfile


# ==== Path setup ====
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent
pretrained_onnx_models_dir = src_dir.parent / "models" / "pretrained" / "ONNX"
pytorch_models_dir = src_dir.parent / "models" / "pretrained" / "PYTORCH"

os.makedirs(pytorch_models_dir, exist_ok=True)

# ==== Set temporary directory to avoid permission issues ====
temp_dir = tempfile.mkdtemp()
os.environ['TMPDIR'] = temp_dir
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir
print(f"Using temporary directory: {temp_dir}")

# ==== Loop through all ONNX models and convert ====
onnx_files = list(pretrained_onnx_models_dir.glob("*.onnx"))

if not onnx_files:
    print(f"[ERROR] No ONNX models found in {pretrained_onnx_models_dir}")
else:
    print(f"Found {len(onnx_files)} ONNX models to convert")

    for onnx_file in onnx_files:
        torch_model_path = pytorch_models_dir / (onnx_file.stem + ".pt")
        
        print(f"\nConverting {onnx_file.name} → {torch_model_path.name} ...")
        
        try:
            # Load and validate ONNX model
            onnx_model = onnx.load(str(onnx_file))
            onnx.checker.check_model(onnx_model) 
            
            pytorch_model = convert(onnx_model)
            pytorch_model.eval() 
            
            # Save the entire model (not just state_dict)
            torch.save(pytorch_model, torch_model_path)
            print(f"[SUCCESS] Saved PyTorch model at: {torch_model_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to convert {onnx_file.name}: {e}")
            # Continue with next model instead of crashing

print("\nConversion process completed!")