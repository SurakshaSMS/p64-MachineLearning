from pathlib import Path

normal_model = Path("./models/mobile_bangla_smishing_model_epoch40_onnx/model.onnx")
quantized_model = Path("./models/mobile_bangla_smishing_model_onnx_quantized_epoch_40/model.onnx")

normal_size = normal_model.stat().st_size / (1024 * 1024)
quantized_size = quantized_model.stat().st_size / (1024 * 1024)

print(f"Normal ONNX size: {normal_size:.2f} MB")
print(f"Quantized ONNX size: {quantized_size:.2f} MB")
print(f"Reduction: {100 - (quantized_size / normal_size * 100):.2f}%")