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

append_api_condition_to_json(fun_string, file_path, new_doc_str)：向JSON文件中添加API条件

get_api_conditions(fun_string, file_path)：获取JSON文件中的api_conditions

append_filtered_combinations_to_json(path, fun_string, new_data)：向JSON文件中添加过滤后的参数组合

add_log(log)：打印日志到控制台和文件
'''

#根据函数名获取函数的文档字符串
def get_doc(function_name):
    try:
        function = eval(function_name)
        return function.__doc__
    except (AttributeError, ImportError, NameError) as e:
        return False


#根据函数文档获取参数列表
#针对torch函数文档进行处理
def extract_parameters_torch(api_doc):

    # 使用正则表达式匹配第一个括号内的内容（参数部分）
    match = re.search(r'\((.*?)\)', api_doc)
    param_str = match.group(1)

    # 处理参数字符串
    parameters = [p.strip().split('=')[0] for p in param_str.split(',')]
    for i in parameters:
        if i == '*':
            parameters.remove(i)
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
def filter_combinations(combinations, conditions):
 
    # 获取条件
    mandatory_params = conditions.get('Mandatory Parameters', [])
    exclusive_groups = conditions.get('Mutually Exclusive Parameter Pairs', [])
    coexistence_groups = conditions.get('Mandatory Coexistence Parameters', [])

    filtered = []
    
    for combo in combinations:
        # 1. 检查是否包含所有必须参数
        if mandatory_params:
            if mandatory_params and not all(param in combo for param in mandatory_params):
                continue

            
        # 2. 检查是否不包含任何互斥参数组中的全部参数
        def filter_exclusive_combinations(param_combinations, exclusive_pairs):
            param_set = set(param_combinations)
            for pair in exclusive_pairs:
                if all(p in param_set for p in pair):
                    return False
            return True
        if not filter_exclusive_combinations(combo, exclusive_groups):
            continue

        # 3. 检查是否满足所有必须共存的参数组
        # 对于每个共存组，检查组合中是否至少包含该组中的一个参数
        # 如果共存组为空，则跳过此检查
        meets_coexistence = all(
            all(param in combo for param in group)
            for group in coexistence_groups
            )
        
        if not meets_coexistence:
            continue
        
        filtered.append(combo)
    
    return filtered

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

# 向JSON文件中添加API条件
def append_api_condition_to_json(path, fun_string, new_data):
    if not new_data:
        condition_dict = {}
    else:
        try:
            # 把字符串解析为 Python 字典
            condition_dict = json.loads(new_data)
        except json.JSONDecodeError as e:
            add_log(f"JSON 解析错误: {e}")
            add_log(f"API 条件数据: {new_data}")
            return

    # 读取原始 JSON 文件内容（如果存在）
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # 添加或更新项
    data[fun_string] = condition_dict

    # 写回 JSON 文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 获取JSON文件中的api_conditions
def get_api_conditions(fun_string, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 直接获取指定函数名对应的条件字典
        return data.get(fun_string, None)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        add_log(f"Error reading file: {e}")
        return None

# 记录log
def add_log(log):
    with open(f'/tmp/Momo_test/{lib_name}_log.txt', "a", encoding="utf-8") as f:
    #with open(r'C:\Users\86184\Desktop\torch_log.txt', "a", encoding="utf-8") as f:
        print(log)  # 打印到控制台
        print(log, file=f)  # 写入文件

# 将过滤好的参数组合写入JSON文件
def append_filtered_combinations_to_json(path, fun_string, new_data):
    # 如果文件存在，加载内容；否则创建空字典
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # 更新或添加新数据
    data[fun_string] = new_data

    # 写入到文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 手动处理output
def handle_output(text: str, model_path: str):
    if model_path == "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct":
        target = "  6.Notions:\n    Only output the json content of the example in the output format, do not add explanations.assistant\n"
        start_index = text.find(target) + len(target)
        json_content = text[start_index:].strip()
        try:
            return json_content
        except json.JSONDecodeError as e:
            return None
    if model_path == "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B":
        end_tag = "</think>"
        if end_tag not in text:
            add_log("未找到 </think> 标签")
            return None

        # 获取 </think> 后的内容
        after_think = text.split(end_tag, 1)[1].strip()

        try:
            return after_think
        except json.JSONDecodeError as e:
            return None
    
