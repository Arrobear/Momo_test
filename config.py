import torch
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import itertools

# 参数设置
checkpoint = "bigcode/starcoder2-15b"
device = "cuda" # for GPU usage or "cpu" for CPU usage

lib_name = "torch"  # 库名称