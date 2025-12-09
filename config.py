import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM, Starcoder2ForCausalLM, BitsAndBytesConfig
import json
import re
import itertools
import os
import bitsandbytes
import sys 
from accelerate import infer_auto_device_map, init_empty_weights
from torch_samename import *
from pathlib import Path
import yaml
import importlib
import inspect
import types
import ast
import textwrap
import requests
import subprocess
import time
import uuid
import random
# from cpgqls_client import *
import gc
import psutil
import traceback
from z3 import *

# 参数设置
# model_path = "/nasdata/haoyahui/Model/starcoder2-15b"
# model_path = "/nasdata/haoyahui/Model/codellama-13b"
model_path = "/home/chaoni/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
# model_path = "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct"
# model_path = "D:/Model/DeepSeek-R1-Distill-Qwen-1.5B"

lib_name = "torch"  # 库名称
# test.cpp
# joern_project  = "pytorch-2.5.1" # joern 项目名
joern_project  = "pytorch-2.5.1" # joern 项目名
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