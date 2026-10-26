import torch,sys,numpy as np
from torchvision.models import resnet18,ResNet18_Weights
from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantType
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from src.predict import load
from pathlib import Path
torch.set_num_threads(4)
m,t,h,c=load()
torch.onnx.export(m,torch.zeros(1,3,224,224),str(Path(__file__).resolve().parents[1]/'.cache/resnet.onnx'),input_names=['image'],output_names=['features'],opset_version=17,dynamo=False)
class Reader(CalibrationDataReader):
 def __init__(self):
  import json,pandas as pd
  from PIL import Image
  root=Path(__file__).resolve().parents[1];rows=pd.read_csv(root/'data/manifest.csv');ids=json.load(open(root/'data/splits.json'))['train'][:32]
  self.items=iter([{'image':t(Image.open(root/'.cache/images'/(rows.iloc[i].sha256+'.img')).convert('RGB')).unsqueeze(0).numpy()} for i in ids])
 def get_next(self):return next(self.items,None)
quantize_static(str(Path(__file__).resolve().parents[1]/'.cache/resnet.onnx'),str(Path(__file__).resolve().parents[1]/'models/resnet-int8.onnx'),Reader(),activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8)
print(Path(str(Path(__file__).resolve().parents[1]/'models/resnet-int8.onnx')).stat().st_size)
