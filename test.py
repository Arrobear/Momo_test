from config import *
from stage_1_function import *
from generate_prompt import *
from generate_prompt import *
from stage_2_function import *


# 示例  
def parse_scala_list(scala_output: str):

    # 提取 List(...) 中的内容
    items = re.findall(r'"(.*?)"', scala_output)
    return items



joern = JoernShell("C:/Users/86184/Desktop/joern-cli/joern.bat")
cpp_func_name = "conv1d_symint"

query = open_query("pytorch-2.5.1")
joern.send_command(query)

torch_cheack = joern.send_command(f"cpg.method.name(\"{cpp_func_name}\").call.name(\"TORCH_CHECK\").argument.order(1).code.l")
torch_contorl = joern.send_command(f"cpg.method.name(\"{cpp_func_name}\").controlStructure.filterNot(_.controlStructureType == \"SWITCH\").condition.code.l")
print("torch_control_str:", type(torch_cheack), len(torch_cheack), torch_cheack)  
print("torch_control_str:", type(torch_contorl), len(torch_contorl), torch_contorl)  

parsed_list = parse_scala_list(torch_cheack)
print(parsed_list)
parsed_list = parse_scala_list(torch_contorl)
print(parsed_list)



# query_1_forhalf= f"cpg.method.name(\"{c_api_1}\")"
# query_1_backhalf = r""".ast.isControlStructure.filter(_.code.startsWith("switch")).foreach { sw => val cond = sw.code.split("\\(")(1).split("\\)")(0).trim; val cases = sw.astChildren.flatMap(_.astChildren).filter(n => n.code.startsWith("case") || n.code.startsWith("default")).toSeq.map(n => if (n.code.startsWith("case")) n.code.split(":")(0).replace("case","").trim else "default"); println(cond + "->" + cases.mkString(",")) }"""

# c_method_info = joern.send_command(query_1_forhalf + query_1_backhalf)

# print(c_method_info)

joern.send_command("exit")   # 退出 Joern

# # 如果 bat 文件需要手动 exit，添加 exit
 
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



def log_analysis():
    with open('conditions/ds_tf_log.txt', 'r', encoding='utf-8') as file:
        log_list = [line.rstrip('\n') for line in file.readlines()]
    for i in range(len(log_list)):
        print(log_list[i])
        if i > 10:
            break   
      

def test_api_doc():
    with open(f"./documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    a = "tf.queue.QueueBase"
    b = "torch.stack"
    waiting_check = []
    abandoned_apis = []
    numpy_redefined_apis = []
    if a == "0":
        api_names = read_file(f"./documentation/{lib_name}_APIdef.txt")

        for i in range(len(api_names)):

            doc = get_doc(api_names[i])
            local_add_log(api_defs[i])
            local_add_log(doc)
            i+=1
            local_add_log(str(i) + '/' + str(len(api_names)))
            if doc is not None and doc is not False and len(doc)<250:
                if "DEPRECATED" in doc:
                    abandoned_apis.append(i)
                elif "See the NumPy" in doc:
                    numpy_redefined_apis.append(i)
                else:
                    waiting_check.append(i)
        local_add_log(f"abandoned_apis: {abandoned_apis}")
        local_add_log(f"numpy_redefined_apis: {numpy_redefined_apis}")
        local_add_log(f"waiting_check: {waiting_check}")
    elif a == "1":
        print(get_doc(b))
    else:
        print(eval(a).__doc__)


def find_all_empty_dicts_in_json(file_path, target_key=None):
    """
    查找 JSON 文件中所有值为空字典 {} 的项，并返回它们的路径
    
    参数:
        file_path (str): JSON 文件路径
        target_key (str, optional): 指定要检查的键名。如果为 None，则检查整个文件
    
    返回:
        list: 包含所有空字典路径的列表，格式为 [{"path": "a.b.c", "value": {}}, ...]
              如果没有空字典，返回空列表 []
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        return []
    
    empty_dicts = []
    
    def _scan_for_empty_dicts(obj, current_path=""):
        if isinstance(obj, dict):
            if obj == {}:
                empty_dicts.append({"path": current_path, "value": {}})
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                _scan_for_empty_dicts(value, new_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                new_path = f"{current_path}[{idx}]"
                _scan_for_empty_dicts(item, new_path)
    
    if target_key is not None:
        if target_key not in data:
            print(f"键 '{target_key}' 不存在")
            return []
        _scan_for_empty_dicts(data[target_key], target_key)
    else:
        _scan_for_empty_dicts(data)
    
    return empty_dicts
    
# message = find_all_empty_dicts_in_json('C:/Users/86184/Desktop/torch_conditions.json')
# for item in message:
#     print(f"空字典路径: {item['path']}, 值: {item['value']}")