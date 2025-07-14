import torch
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import itertools
import os

# 参数设置
checkpoint = "bigcode/starcoder2-15b"
# checkpoint = "bigcode/starcoder2-3b"
model_path = "/nasdata/haoyahui/Model/starcoder2-15b"

torch.cuda.set_device(0)  # 指定默认使用 GPU 0

device = torch.device("cuda:3") 

lib_name = "torch"  # 库名称
