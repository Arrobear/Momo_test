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


# Runtime settings may be supplied by rq3.run or directly through the environment.
root_path = os.environ.get(
    "MOMO_ROOT_PATH", str(Path(__file__).resolve().parent.parent)
)

# 将被测库的源码目录加到 sys.path，确保 pip install 失败时也能 import 被测库
_dl_lib_dir = os.environ.get(
    "MOMO_REPOSITORY_ROOT", os.path.join(root_path, "dl_lib")
)
if os.path.isdir(_dl_lib_dir):
    for _entry in os.listdir(_dl_lib_dir):
        _candidate = os.path.join(_dl_lib_dir, _entry)
        if os.path.isdir(_candidate) and _candidate not in sys.path:
            sys.path.insert(0, _candidate)
del _dl_lib_dir

API_KEY = os.environ.get("MOMO_API_KEY", "")
BASE_URL = os.environ.get("MOMO_BASE_URL", "https://www.su8.codes/v1")
MODEL = os.environ.get("MOMO_MODEL", "gpt-5.5")

def make_client():
    from openai import OpenAI
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        default_headers={"User-Agent": "python-httpx/0.28.1"},
    )


model_path = os.environ.get(
    "MOMO_LOCAL_MODEL_PATH",
    str(Path(root_path) / "models" / "DeepSeek-R1-Distill-Qwen-32B"),
)

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

lib_name = os.environ.get("MOMO_LIB_NAME", "ansible")  # 库名称
lib_gitname = os.environ.get("MOMO_LIB_GITNAME", "ansible")
_reference_commit = os.environ.get("MOMO_REFERENCE_COMMIT")
_candidate_commit = os.environ.get("MOMO_CANDIDATE_COMMIT")
if bool(_reference_commit) != bool(_candidate_commit):
    raise ValueError("MOMO_REFERENCE_COMMIT and MOMO_CANDIDATE_COMMIT must be set together")
test_version = (
    [_reference_commit, _candidate_commit]
    if _reference_commit and _candidate_commit
    else ["ea6ba08", "334db14"]
)

# test.cpp
# joern_project  = "pytorch-2.5.1" # joern 项目名
# {lib_name}_{commit_hash}
conmmit_hash = test_version[0]
joern_project = "ansible_6"  # Retain the assignment expected by run_momo_new.py.
joern_project = os.environ.get("MOMO_JOERN_PROJECT", joern_project)
joern_bat_path = os.environ.get("MOMO_JOERN_PATH", "C:/joern-cli/joern.bat")

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
