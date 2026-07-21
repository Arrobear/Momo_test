# import torch
# import tensorflowY as tf
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, Starcoder2ForCausalLM, BitsAndBytesConfig
except ImportError:
    AutoTokenizer = AutoModelForCausalLM = Starcoder2ForCausalLM = BitsAndBytesConfig = None
import json
import re
import itertools
import os
import sys
try:
    from accelerate import infer_auto_device_map, init_empty_weights
except ImportError:
    infer_auto_device_map = init_empty_weights = None
from torch_samename import *
from pathlib import Path
import yaml
import importlib
import inspect
import ast
import textwrap
import subprocess
import time
import uuid
import gc
import psutil
import traceback
from itertools import product
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import json_repair
except ImportError:
    json_repair = None
import math
import random


# root_path = "/data/chaoni/haoyahui"
root_path = "C:/Users/86184/Desktop/Papers"

# 将被测库的源码目录加到 sys.path，确保 pip install 失败时也能 import 被测库
_dl_lib_dir = os.path.join(root_path, "documentation", "dl_lib")
if os.path.isdir(_dl_lib_dir):
    for _entry in os.listdir(_dl_lib_dir):
        _candidate = os.path.join(_dl_lib_dir, _entry)
        if os.path.isdir(_candidate) and _candidate not in sys.path:
            sys.path.insert(0, _candidate)
del _dl_lib_dir, _entry, _candidate

API_KEY = "su8-8e384b5f169adcf570dd40aa9b"
BASE_URL = "https://www.su8.codes/v1"
MODEL = "gpt-5.5"

def make_client():
    from openai import OpenAI
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        default_headers={"User-Agent": "python-httpx/0.28.1"},
    )


# 参数设置
# model_path = "/nasdata/haoyahui/Model/starcoder2-15b"
# model_path = "/nasdata/haoyahui/Model/codellama-13b"
model_path = root_path + "/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
# model_path = "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct"
# model_path = "D:/Model/DeepSeek-R1-Distill-Qwen-1.5B"

try:
    import glom
except ImportError:
    glom = None
try:
    import boolean
except ImportError:
    boolean = None
try:
    import mimesis
except ImportError:
    mimesis = None
try:
    import parse
except ImportError:
    parse = None


USE_SOURCE_RESOLVER = True  # True: AST方式读源码 / False: 原importlib方式

lib_name = "ansible"  # 库名称
lib_gitname = "ansible"
test_version = ['ea6ba08','334db14']

# test.cpp
# joern_project  = "pytorch-2.5.1" # joern 项目名
# {lib_name}_{commit_hash}
conmmit_hash = "aba793e7"
joern_project  = "ansible_6" # joern 项目名
joern_bat_path = "C:/Users/86184/Desktop/joern-cli/joern.bat"

gpu_str = os.environ.get("FREE_GPUS", "")

gpu_ids = [int(g) for g in gpu_str.strip().split()] if gpu_str else []


torch_samename_list = [
    "torch.dequantize",
    "torch.where",
    "torch.normal",
    "torch.max",
    "torch.pow",
    "torch.argmax",
    "torch.all",
    "torch.any",
    "torch.min",
    "torch.mean",
    "torch.median",
    "torch.nanmedian",
    "torch.nansum",
    "torch.prod",
    "torch.sum"
]

large_combination_list = ['tf.keras.optimizers.Adadelta',
                           'tf.keras.layers.DepthwiseConv2D', 
                           'tf.nn.sampled_softmax_loss', 
                           'tf.keras.layers.SimpleRNN', 
                           'tf.keras.layers.LayerNormalization', 
                           'tf.keras.optimizers.Nadam', 
                           'tf.keras.layers.GRUCell',
                             'tf.keras.mixed_precision.LossScaleOptimizer', 
                             'tf.keras.layers.Embedding', 
                             'tf.keras.layers.Conv3DTranspose', 
                             'tf.keras.optimizers.RMSprop', 
                             'tf.keras.layers.GRU', 
                             'tf.data.experimental.make_csv_dataset', 
                             'tf.keras.applications.MobileNetV3Small', 
                             'tf.keras.layers.Conv2D', 
                             'tf.keras.Model', 
                             'tf.keras.layers.Dense', 
                             'tf.keras.layers.Conv1D', 
                             'tf.keras.optimizers.Adamax', 
                             'tf.keras.optimizers.Adam', 
                             'tf.keras.applications.MobileNetV3Large', 
                             'tf.keras.layers.Conv2DTranspose', 
                             'tf.keras.layers.LSTMCell', 
                             'tf.io.encode_jpeg', 
                             'tf.keras.layers.ConvLSTM2D', 
                             'tf.keras.layers.MultiHeadAttention', 
                             'tf.keras.layers.SimpleRNNCell', 
                             'tf.keras.layers.BatchNormalization', 
                             'tf.keras.layers.SeparableConv1D', 
                             'tf.keras.utils.get_file', 
                             'tf.random.fixed_unigram_candidate_sampler', 
                             'tf.keras.layers.Conv1DTranspose', 
                             'tf.keras.layers.LSTM', 
                             'tf.data.experimental.make_batched_features_dataset', 
                             'tf.strided_slice']

factory_names = {
    "zeros", "ones", "empty", "full",
    "rand", "randn", "rand_like", "randn_like",
    "eye", "arange", "linspace", "logspace",
    "tensor", "as_tensor"
}
