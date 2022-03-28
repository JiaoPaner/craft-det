# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 14:38
# @Author  : jiaopaner
import sys
sys.path.insert(0, './')
import torch
from collections import OrderedDict
from craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load("/Volumes/storage/resources/models/paddle-ocr-models/craft_mlt_25k.pth", map_location="cpu")))
    net.eval()


    #dynamic shape
    x = torch.randn((1, 3, 960, 960))

    torch.onnx.export(net, x, './pd_model/model.onnx', opset_version=11, input_names=["input"],
                      output_names=["output"], dynamic_axes={'input': [2,3]})

    # x2paddle --framework=onnx --model=./pd_model/model.onnx --save_dir=pd_model_dynamic