import onnx

model = onnx.load("/root/Robot_pet/face_ws/face_recog/models/onnx/version-RFB-320.onnx")

for input_tensor in model.graph.input:
    print(f"Input Name: {input_tensor.name}")
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input Shape: {shape}")