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

# 参数设置
# model_path = "/nasdata/haoyahui/Model/starcoder2-15b"
# model_path = "/nasdata/haoyahui/Model/codellama-13b"
model_path = "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
# model_path = "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct"
# model_path = "D:/Model/DeepSeek-R1-Distill-Qwen-1.5B"

lib_name = "torch"  # 库名称

gpu_str = os.environ.get("FREE_GPUS", "")

gpu_ids = [int(g) for g in gpu_str.strip().split()] if gpu_str else []

torch_samename_list = [
    "torch.dequantize",
    "torch.where",
    "torch.normal",
    "torch.add",
    "torch.max",
    "torch.mul",
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
    "torch.std",
    "torch.std_mean",
    "torch.sum",
    "torch.var",
    "torch.var_mean",
    "torch.repeat_interleave",
    "torch.trapz"

]

ds_torch_fail_list = [
    "torch.nn.functional.embedding_bag",
    "torch.nn.functional.grid_sample",
    "torch.sparse_coo_tensor",
    "torch.load",
    "torch.unique",
    "torch.stft",
    "torch.einsum",
    "torch.triu",
    "torch.svd",
    "torch.lobpcg",
    "torch.use_deterministic_algorithms",
    "torch.nn.Conv1d",
    "torch.nn.Conv2d",
    "torch.nn.Conv3d",
    "torch.nn.ConvTranspose2d",
    "torch.nn.ConvTranspose3d",
    "torch.nn.Unfold",
    "torch.nn.SyncBatchNorm",
    "torch.nn.RNN",
    "torch.nn.LSTM",
    "torch.nn.GRU",
    "torch.nn.Embedding",
    "torch.nn.EmbeddingBag",
    "torch.nn.CrossEntropyLoss",
    "torch.nn.NLLLoss",
    "torch.nn.CTCLoss",
    "torch.nn.BCEWithLogitsLoss",
    "torch.nn.TripletMarginWithDistanceLoss",
    "torch.nn.Upsample",
    "torch.nn.parallel.DistributedDataParallel",
    "torch.nn.modules.lazy.LazyModuleMixin"
]

