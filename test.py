from config import *
from stage_1_function import *
from generate_prompt import *
from generate_prompt import *
from stage_2_function import *

# constraints = [
#                     "input.shape[1] == weight.shape[1]",
#                     "input.dtype == weight.dtype"
#                 ]

# combo = {
#                     "input": "(torch.randn((8, 16, 256,), dtype=torch.float32) + 1j * torch.randn((8, 16, 256,), dtype=torch.float32)).to(dtype=torch.complex64)",
#                     "weight": "(torch.randn((16, 16, 256,), dtype=torch.float32) + 1j * torch.randn((16, 16, 256,), dtype=torch.float32)).to(dtype=torch.complex64)"
#                 }


# print(check_constraints(combo, constraints))










api_names = read_file(f"/home/chaoni/haoyahui/documentation/{lib_name}_APIdef.txt")
length_api_names = len(api_names)
k = 0
for i in range(length_api_names):
    api_inputs = []
    print(i)
    api_name = api_names[i]
    arg_combinations = read_json_api(api_name=api_name, file_path=f"/home/chaoni/haoyahui/documentation/arg_combinations/", read_mode="combination")
    # api_code = read_json_api(api_name=api_name, file_path=f"../documentation/api_src_code/", read_mode="src_code")
    # error_combinations = read_json_api(api_name=api_name, file_path=f"../documentation/error_combinations/", read_mode="error_combination")
    # conditions = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
    arg_spaces = read_json_api(api_name=api_name, file_path=f"/home/chaoni/haoyahui/documentation/arg_space/", read_mode="arg_space")

    k = k + len(arg_combinations) * len(arg_spaces)
    print(k)
# print(k)
# api_name =  "torch.index_select"
# arg_combinations = read_json_api(api_name=api_name, file_path=f"./documentation/arg_combinations/", read_mode="combination")
# api_code = read_json_api(api_name=api_name, file_path=f"./documentation/api_src_code/", read_mode="src_code")
# arg_spaces = read_json_api(api_name=api_name, file_path=f"./documentation/arg_space/", read_mode="arg_space")
# error_combinations = read_json_api(api_name=api_name, file_path=f"./documentation/error_combinations/", read_mode="error_combination")
# conditions = read_json_api(api_name=api_name, file_path=f"./documentation/conditions/", read_mode="conditions")

# for arg_combination in arg_combinations:
#     for arg_space in arg_spaces:
#         prompt = generate_prompt_3(api_name, arg_combination, api_code, arg_space, conditions["Parameter type"])
        
#         print(prompt)
#         break
#     break


# # 循环写入命令
# # for cmd in commands:
# process.stdin.write(query+"\n")  # 写入命令
# process.stdin.flush()
# process.stdin.write(f"cpg.method.name(\"{c_api}\").l\n")  # 强制刷新缓冲区
# process.stdin.flush()
# #process.stdin.write("")  # 强制刷新缓冲区
# #process.stdin.flush()  # 强制刷新缓冲区

# # 获取输出
# stdout, stderr = process.communicate()
# print("输出:\n", stdout)






















# 测试代码区


# 要提取的模块列表（可以根据需要扩展）

# import torch.nn.functional as F
# import torch.linalg
# import torch.fft


# # 模块列表
# modules_to_extract = [
#     ("torch", torch),
#     ("torch.nn.functional", F),
#     ("torch.linalg", torch.linalg),
#     ("torch.fft", torch.fft)
# ]

# apis = []

# def extract_function_names(module, module_name):
#     """提取模块的函数名，不包含参数"""
#     for name, obj in inspect.getmembers(module):
#         if name.startswith("_"):
#             continue  # 忽略私有函数
#         if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
#             apis.append(f"{module_name}.{name}")

# # 提取每个模块的函数名
# for mod_name, mod in modules_to_extract:
#     extract_function_names(mod, mod_name)


# # 去重、排序
# apis = sorted(set(apis))

# # 写入文件
# with open("torch_function_names.txt", "w", encoding="utf-8") as f:
#     for api in apis:
#         f.write(api + "\n")

# print(f"收集到 {len(apis)} 个函数名，已保存到 torch_function_names.txt")


 #测试用函数



# def log_analysis():
#     with open('conditions/ds_tf_log.txt', 'r', encoding='utf-8') as file:
#         log_list = [line.rstrip('\n') for line in file.readlines()]
#     for i in range(len(log_list)):
#         print(log_list[i])
#         if i > 10:
#             break   
      

# def test_api_doc():
#     with open(f"./documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
#         api_defs = [line.strip() for line in file]

#     a = "tf.queue.QueueBase"
#     b = "torch.stack"
#     waiting_check = []
#     abandoned_apis = []
#     numpy_redefined_apis = []
#     if a == "0":
#         api_names = read_file(f"./documentation/{lib_name}_APIdef.txt")

#         for i in range(len(api_names)):

#             doc = get_doc(api_names[i])
#             local_add_log(api_defs[i])
#             local_add_log(doc)
#             i+=1
#             local_add_log(str(i) + '/' + str(len(api_names)))
#             if doc is not None and doc is not False and len(doc)<250:
#                 if "DEPRECATED" in doc:
#                     abandoned_apis.append(i)
#                 elif "See the NumPy" in doc:
#                     numpy_redefined_apis.append(i)
#                 else:
#                     waiting_check.append(i)
#         local_add_log(f"abandoned_apis: {abandoned_apis}")
#         local_add_log(f"numpy_redefined_apis: {numpy_redefined_apis}")
#         local_add_log(f"waiting_check: {waiting_check}")
#     elif a == "1":
#         print(get_doc(b))
#     else:
#         print(eval(a).__doc__)


# def find_all_empty_dicts_in_json(file_path, target_key=None):
#     """
#     查找 JSON 文件中所有值为空字典 {} 的项，并返回它们的路径
    
#     参数:
#         file_path (str): JSON 文件路径
#         target_key (str, optional): 指定要检查的键名。如果为 None，则检查整个文件
    
#     返回:
#         list: 包含所有空字典路径的列表，格式为 [{"path": "a.b.c", "value": {}}, ...]
#               如果没有空字典，返回空列表 []
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"读取文件失败: {str(e)}")
#         return []
    
#     empty_dicts = []
    
#     def _scan_for_empty_dicts(obj, current_path=""):
#         if isinstance(obj, dict):
#             if obj == {}:
#                 empty_dicts.append({"path": current_path, "value": {}})
#             for key, value in obj.items():
#                 new_path = f"{current_path}.{key}" if current_path else key
#                 _scan_for_empty_dicts(value, new_path)
#         elif isinstance(obj, list):
#             for idx, item in enumerate(obj):
#                 new_path = f"{current_path}[{idx}]"
#                 _scan_for_empty_dicts(item, new_path)
    
#     if target_key is not None:
#         if target_key not in data:
#             print(f"键 '{target_key}' 不存在")
#             return []
#         _scan_for_empty_dicts(data[target_key], target_key)
#     else:
#         _scan_for_empty_dicts(data)
    
#     return empty_dicts
    
# message = find_all_empty_dicts_in_json('C:/Users/86184/Desktop/torch_conditions.json')
# for item in message:
#     print(f"空字典路径: {item['path']}, 值: {item['value']}")