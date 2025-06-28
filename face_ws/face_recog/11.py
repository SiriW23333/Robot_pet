import torch
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd

# ���� PyTorch ģ��
num_classes = 2
model = create_Mb_Tiny_RFB_fd(num_classes, is_test=True)
model.load_state_dict(torch.load("models/pretrained/version-RFB-320.pth"))

# ת��Ϊ ONNX ��ʽ
dummy_input = torch.randn(1, 3, 320, 320)  # ����ģ�������С����
torch.onnx.export(model, dummy_input, "version-RFB-320.onnx", verbose=True)
