import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM, Starcoder2ForCausalLM
import json
import re
import itertools
import os
import bitsandbytes
import sys

# 参数设置
# model_path = "/nasdata/haoyahui/Model/starcoder2-15b"
model_path = "/nasdata/haoyahui/Model/codellama-13b"
# model_path = "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
# model_path = "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct"


torch.cuda.set_device(0)  # 指定默认使用 GPU 0

# device = torch.device("cuda:4") 

lib_name = "torch"  # 库名称

