from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

# Original exported ONNX model
input_model_path = Path("./models/mobile_bangla_smishing_model_epoch40_onnx/model.onnx")

# Separate output folders
preprocessed_dir = Path("./models/bangla_smishing_model_onnx_preprocessed_epoch_40")
quantized_dir = Path("./models/mobile_bangla_smishing_model_onnx_quantized_epoch_40")

preprocessed_model_path = preprocessed_dir / "model.onnx"
quantized_model_path = quantized_dir / "model.onnx"

preprocessed_dir.mkdir(parents=True, exist_ok=True)
quantized_dir.mkdir(parents=True, exist_ok=True)

print("Pre-processing ONNX model...")

quant_pre_process(
    input_model_path=str(input_model_path),
    output_model_path=str(preprocessed_model_path),
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=True,
)

print("Quantizing preprocessed ONNX model...")

quantize_dynamic(
    model_input=str(preprocessed_model_path),
    model_output=str(quantized_model_path),
    weight_type=QuantType.QUInt8,
    op_types_to_quantize=["MatMul"],
    per_channel=True,
)

print("Done.")
print(f"Preprocessed model: {preprocessed_model_path}")
print(f"Quantized model: {quantized_model_path}")