from config import *
from function import *
from main import *
from generate_prompt import *

def append_to_txt_file(file_path, content, mode='a', encoding='utf-8'):
    """
    将字符串内容添加到文本文件中
    
    参数:
        file_path (str): 目标文件的路径
        content (str): 要添加的内容
        mode (str): 文件打开模式，默认为'a'(追加)
                   'a' - 追加(文件不存在则创建)
                   'w' - 写入(会覆盖原有内容)
        encoding (str): 文件编码，默认为'utf-8'
    
    返回:
        bool: 操作是否成功
    """
    try:
        with open(file_path, mode, encoding=encoding) as file:
            file.write(content)
            if not content.endswith('\n'):  # 如果内容不以换行符结尾，自动添加
                file.write('\n')
        return True
    except Exception as e:
        print(f"写入文件时出错: {e}")
        return False

# 测试代码区
# with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
#     api_defs = [line.strip() for line in file]
# api_names = read_file(f"{lib_name}_APIdef.txt")
# for i in range(len(api_names)):
#     args = get_all_parameters(api_names[i], api_defs[i])
#     append_to_txt_file(f'C:/Users/86184/Desktop/{lib_name}_args.txt', f"{api_names[i]}: {args}", mode='a', encoding='utf-8')


# json_path = Path(__file__).parent / "conditions" / f"{lib_name}_conditions.json"
# print(get_api_conditions("torch.initial_seed", str(json_path)))


torch.nn.functional.conv1d

# print(extract_invalid_parameter_combinations()[0][1])
# try:
#     with open(r'C:\Users\86184\Desktop\test.txt', 'r', encoding='utf-8') as f:
#         content = f.read()
#         print("文件内容预览（前500字符）:", content[:500])
# except:
#     print("无法读取文件内容")



 #测试用函数



def log_analysis():
    with open('conditions/ds_tf_log.txt', 'r', encoding='utf-8') as file:
        log_list = [line.rstrip('\n') for line in file.readlines()]
    for i in range(len(log_list)):
        print(log_list[i])
        if i > 10:
            break   
      

def test_api_doc():
    with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    a = "tf.queue.QueueBase"
    b = "torch.stack"
    waiting_check = []
    abandoned_apis = []
    numpy_redefined_apis = []
    if a == "0":
        api_names = read_file(f"{lib_name}_APIdef.txt")

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