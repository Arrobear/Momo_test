from config import *
from function import *
from main import *
from generate_prompt import *

# print(torch_samename_data["torch.where_1"])

# def log_analysis():
#     with open('conditions/ds_tf_log.txt', 'r', encoding='utf-8') as file:
#         log_list = [line.rstrip('\n') for line in file.readlines()]
#     for i in range(len(log_list)):
#         print(log_list[i])
#         if i > 10:
#             break   
# log_analysis()


# with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
#     api_defs = [line.strip() for line in file]

# a = "0"
# b = "torch.stack"
# k = []
# if a == "0":
#     api_names = read_file(f"{lib_name}_APIdef.txt")

#     for i in range(len(api_names)):

#         doc = get_doc(api_names[i])
#         local_add_log(api_defs[i])
#         local_add_log(doc)
#         i+=1
#         local_add_log(str(i) + '/' + str(len(api_names)))
#         if doc is not None and doc is not False and len(doc)<250:
#             k.append(i)
#     local_add_log(k)
# elif a == "1":
#     print(get_doc(b))
# else:
#     print(eval(a).__doc__)



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
    
message = find_all_empty_dicts_in_json('C:/Users/86184/Desktop/torch_conditions.json')
for item in message:
    print(f"空字典路径: {item['path']}, 值: {item['value']}")