import torch
import tensorflow as tf
import inspect




extract_lib = "tf"

save_path = f"{extract_lib}_function_names.txt"

# 模块列表，可根据需要扩展
torch_modules_to_extract = [
    ("torch", torch),
    ("torch.nn.functional", torch.nn.functional),
    ("torch.linalg", torch.linalg),
    ("torch.fft", torch.fft)
]

tf_modules_to_extract = [
    ("tf", tf),
    ("tf.image", tf.image),
    ("tf.signal", tf.signal),
    ("tf.math", tf.math),
    ("tf.linalg", tf.linalg),
    ("tf.random", tf.random),
    ("tf.nn", tf.nn),
    ("tf.keras.activations", tf.keras.activations),
    ("tf.keras.losses", tf.keras.losses),
    ("tf.data.experimental", tf.data.experimental)
]

apis = []

def extract_function_names(module, module_name):
    """提取模块的函数名，不包含参数"""
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue  # 忽略私有函数
        if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
            apis.append(f"{module_name}.{name}")

# 提取每个模块的函数名
for mod_name, mod in eval(f'{extract_lib}_modules_to_extract'):
    extract_function_names(mod, mod_name)

# 去重、排序
apis = sorted(set(apis))

# 写入文件
with open(save_path, "w", encoding="utf-8") as f:
    for api in apis:
        f.write(api + "\n")

print(f"收集到 {len(apis)} 个 函数名，已保存到 {save_path}")

