import onnx

model = onnx.load("./models/mobile_bangla_smishing_model_onnx_quantized_epoch_40/model.onnx")

print("Inputs:")
for input_item in model.graph.input:
    print(input_item.name)

print("Outputs:")
for output_item in model.graph.output:
    print(output_item.name)