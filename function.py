from config import *

'''
**该文件内存储完成各种基本操作的函数**

包括：

get_doc(function_name)：根据函数名获取函数的文档字符串

extract_parameters_torch(api_doc)：根据torch函数文档获取参数列表

extract_parameters_tf(api_doc)：根据tf函数文档获取参数列表

generate_all_combinations(args)：获取所有参数的组合

filter_combinations(combinations, condition)：过滤不合法的参数组合

read_file(file_path)：读取文件

append_api_condition_to_json(fun_string, file_path, new_doc_str)：向JSON文件中添加API文档



'''

#根据函数名获取函数的文档字符串
def get_doc(function_name):
    function = eval(function_name)
    return function.__doc__

#根据函数文档获取参数列表
#针对torch函数文档进行处理
def extract_parameters_torch(api_doc):

    # 使用正则表达式匹配第一个括号内的内容（参数部分）
    match = re.search(r'\((.*?)\)', api_doc)
    param_str = match.group(1)

    # 处理参数字符串
    parameters = [p.strip().split('=')[0] for p in param_str.split(',')]
    
    return parameters

#针对tf函数文档进行处理
def extract_parameters_tf(api_doc):
    # 使用正则表达式匹配Args部分的所有参数
    #tf↓
    #pattern = r'Args:\n(.*?)(?=\n\n|\n\w+:|$)'
    #torch↓
    pattern = r'Args:\n(.*?)(?=\n\w+:|Returns:|$)'
    args_section = re.search(pattern, api_doc, re.DOTALL)
    
    if not args_section:
        return []
    
    # 提取每个参数行
    param_lines = args_section.group(1).split('\n')
    #for i in param_lines:
        #print(i)

    parameters = []
    
    for line in param_lines:
        # 匹配参数名（第一个冒号前的单词）
        param_match = re.match(r'^\s*(\w+)\s*:', line.strip())
        if param_match:
            parameters.append(param_match.group(1))
    
    return parameters

#获取所有参数的组合
def generate_all_combinations(args):
    all_combinations = []
    for r in range(1, len(args) + 1):
        combinations = itertools.combinations(args, r)
        all_combinations.extend([list(comb) for comb in combinations])
    return all_combinations

#过滤不合法的参数组合
def filter_combinations(combinations, condition):
    filtered_combinations = []
    for comb in combinations:
        if all(p in comb for p in condition["Mandatory Parameters"]):
            if not any(p in comb for p in condition["Mandatory Coexistence Parameters"]):
                if not any(p in comb for p in condition["Mutually Exclusive Parameter Pairs"]):
                    filtered_combinations.append(comb)
    return filtered_combinations

#读取文件
def read_file(file_path):
    api_names = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    for i in lines:
        pattern = r"^[^(]*"
        match = re.match(pattern, i)
        api_names.append(match.group() if match else None)

    return api_names

# 向JSON文件中添加API文档
def append_api_condition_to_json(fun_string, file_path, new_doc_str):

    # 提取JSON部分
    try:
        # 找到JSON部分的起始和结束位置
        start_idx = new_doc_str.find('{')
        end_idx = new_doc_str.rfind('}') + 1
        json_str = new_doc_str[start_idx:end_idx]
        
        # 解析JSON
        new_data = json.loads(json_str)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"无法从字符串中提取有效JSON: {e}")

    # 读取或初始化现有数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 文件不存在或为空，初始化为包含API文档的列表
        existing_data = {"api_conditions": []}

    # 确保数据结构正确
    if not isinstance(existing_data, dict) or "api_conditions" not in existing_data:
        existing_data = {"api_conditions": []}

    # 添加新文档（带时间戳）
    doc_entry = {
        fun_string: new_data
    }
    existing_data["api_conditions"].append(doc_entry)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)