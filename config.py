import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import itertools
import os
import bitsandbytes


# 参数设置
#checkpoint = "bigcode/starcoder2-15b"
# checkpoint = "bigcode/starcoder2-3b"
# model_path = "/nasdata/haoyahui/Model/starcoder2-15b"
# model_path = "/nasdata/haoyahui/Model/codellama-13b"
# model_path = "/nasdata/haoyahui/Model/deepseek-coder-6.7b-base"
# model_path = "/nasdata/haoyahui/Model/CodeLlama-70b-Instruct-hf"
# model_path = "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
model_path = "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct"


torch.cuda.set_device(0)  # 指定默认使用 GPU 0

device = torch.device("cuda:0") 

lib_name = "torch"  # 库名称
