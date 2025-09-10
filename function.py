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

def filter_apidocument(api_doc):
    # 定义正则表达式模式，匹配See :class:`~到` for more details.之间的内容
    if lib_name == "torch":
        pattern_0 = r':class:`~(.*?)` for more'
        match_0 = re.search(pattern_0, api_doc)

        pattern_1 = r'See :class:`~(.*?)`'
        match_1 = re.search(pattern_1, api_doc)

        pattern_2 = r'See :class:`(.*?)` for details'
        match_2 = re.search(pattern_2, api_doc)

        pattern_3 = r'Alias of :func:`(.*?)`'
        match_3 = re.search(pattern_3, api_doc)

        pattern_4 = r'of :meth:`(.*?)`'
        match_4 = re.search(pattern_4, api_doc)

        pattern_5 = r'Alias for :func:`(.*?)`'
        match_5 = re.search(pattern_5, api_doc)

        if match_0:
            return match_0.group(1)  # 返回捕获组中的内容
        elif match_1:
            return match_1.group(1)
        elif match_2:
            return match_2.group(1)
        elif match_3:
            return match_3.group(1)
        elif match_4:
            return match_4.group(1)
        elif match_5:
            return match_5.group(1)
        return None  # 如果没有匹配到，返回None
    elif lib_name == "tf":
        pattern_0 = r':class:`~(.*?)` for more'
        match_0 = re.search(pattern_0, api_doc)



#根据函数名获取函数的文档字符串
def get_doc(function_name):
    
    if lib_name == "torch":
        if function_name in torch_samename_data:
            return torch_samename_data[function_name]
        
        if function_name in ["torch.scatter", "torch.scatter_add"]:
            return eval(filter_apidocument(eval(function_name).__doc__)).__doc__
        if function_name.endswith("_"):
            function_name_ = function_name[:-1]
            try:
                function = eval(function_name)
                api_doc_1 = function.__doc__
            except (AttributeError, ImportError, NameError) as e:
                return False
            try:
                function = eval(function_name_)
                api_doc_2 = function.__doc__
            except (AttributeError, ImportError, NameError) as e:
                return False
            
            return api_doc_1 + '\n' + api_doc_2 + '\n' + get_doc(function_name_)
        
        try:
            function = eval(function_name)
            api_doc = function.__doc__
            if api_doc is None:
                return False
    
        except (AttributeError, ImportError, NameError) as e:
            return False
        hash_list = ["Args:" in api_doc , 
                     "math::" in api_doc, 
                     "Shape:" in api_doc , 
                     "Arguments:" in api_doc , 
                     "-> torch.dtype" in api_doc , 
                     "from_numpy(ndarray)" in api_doc , 
                     "torch.moveaxis" in api_doc , 
                     "Examples:" in api_doc ,
                     function_name == "torch.seed",
                     function_name =="torch.initial_seed",
                     function_name =="torch.get_rng_state",
                     function_name =="torch.get_num_threads",
                     function_name =="torch.get_num_interop_threads",
                     function_name =="torch.compiled_with_cxx11_abi",
                     function_name =="torch.are_deterministic_algorithms_enabled"
                     ]

        if True in hash_list:
            return api_doc
        else:
            func_name = filter_apidocument(api_doc)
            if func_name is None:
                return api_doc
            return get_doc(func_name)          
        
    elif lib_name == "tf":
        try:
            function = eval(function_name)
            api_doc = function.__doc__
        except (AttributeError, ImportError, NameError) as e:
            return False

        if api_doc is None:
            return False
        
        return api_doc


#根据函数文档获取参数列表
#针对torch函数文档进行处理
def extract_parameters_torch(api_doc, api_def):
    
    if len(api_doc) > len(api_def):
        new_api_doc = api_doc[:len(api_def)+100]
    else:
        new_api_doc = api_doc
    # 使用正则表达式匹配第一个括号内的内容（参数部分）
    match = re.search(r'\((.*?)\)', new_api_doc)
    
    if not match:
        match_1 = re.search(r'\((.*?)\)', api_def)
        param_str = match_1.group(1)
        # 处理参数字符串
        parameters = [p.strip().split('=')[0] for p in param_str.split(',')]
        for i in parameters:
            if i == '*':
                parameters.remove(i)
        return parameters
    else:
        param_str = match.group(1)
        # 处理参数字符串
        parameters = [p.strip().split('=')[0] for p in param_str.split(',')]
        for i in parameters:
            if i == '*':
                parameters.remove(i)
        return parameters

#针对tf函数文档进行处理
def extract_parameters_tf(api_doc, api_def):
    # 使用正则表达式匹配Args部分的所有参数
    #tf↓
    #pattern = r'Args:\n(.*?)(?=\n\n|\n\w+:|$)'
    #torch↓
    if "Args:" in api_doc:
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
    else:
        # 如果没有找到Args部分，使用api_def获取参数列表
        param_str = api_def.split('(')[1].split(')')[0]
        parameters = [p.strip().split('=')[0] for p in param_str.split(',')]
        for i in parameters:
            if i == '*':
                parameters.remove(i)
        return parameters

#获取函数所有合法参数
def get_all_parameters(api_name: str):
    json_filename = f"{lib_name}_conditions.json"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "conditions", json_filename)
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if api_name not in data:
        raise KeyError(f"API '{api_name}' not found in JSON file")
    
    if "Parameter type" not in data[api_name]:
        return []
    
    return list(data[api_name]["Parameter type"].keys())
    # api_doc = get_doc(fun_string)
    # 先根据api_doc获取参数列表
    # 如果不能通过api_doc获取参数列表，则使用api_def获取参数列表

    # if lib_name == "torch":
    #     return extract_parameters_torch(api_doc, api_def)
    # elif lib_name == "tf":
    #     return extract_parameters_tf(api_doc, api_def)
    # 选择对应的参数列表提取方法提取参数参数列表


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
def add_log(path, log):
    #with open(f'/tmp/Momo_test/{lib_name}_log.txt', "a", encoding="utf-8") as f:
    # with open(r'C:\Users\86184\Desktop\torch_log.txt', "a", encoding="utf-8") as f:
    file_path = path
    
    # 确保目录和文件都存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 写入日志（如果文件不存在会自动创建）
    with open(file_path, "a", encoding="utf-8") as f:
        print(log)  # 打印到控制台
        print(log, file=f)  # 写入文件

# 记录log
def local_add_log(log):
    # with open(f'/tmp/Momo_test/{lib_name}_log.txt', "a", encoding="utf-8") as f:
    with open(f'C:/Users/86184/Desktop/local_{lib_name}_filter_log.txt', "a", encoding="utf-8") as f:
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


# 读取JSON文件中的过滤好的参数组合
def is_file_too_large(file_path, max_size_mb=10):
    """
    检查文件是否过大
    
    参数:
    file_path (str): 文件路径
    max_size_mb (float): 最大允许的文件大小（MB），默认10MB
    
    返回:
    bool: 如果文件超过指定大小返回True，否则返回False
    """
    try:
        if not os.path.exists(file_path):
            return False
            
        file_size = os.path.getsize(file_path)  # 字节数
        file_size_mb = file_size / (1024 * 1024)  # 转换为MB
        
        return file_size_mb > max_size_mb
        
    except Exception as e:
        print(f"检查文件大小时发生错误：{e}")
        return False



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
        for i in range(len(after_think)-1, -1, -1):
            if after_think[i] == '}':
                # 找到最后一个'}'，返回从开头到该位置的子串
                return after_think[:i+1]

        try:
            return after_think
        except json.JSONDecodeError as e:
            return None

# 封装不同模型的输入输出模式 
def generate_input(prompt, tokenizer, model):

    model_path_list = [
        "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct",
        "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
    ]

    if model_path not in model_path_list:
    
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        )
    else:
        inputs = tokenizer.apply_chat_template(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        )
    return inputs

def generate_output(inputs, model, tokenizer):
    model_path_list = [
        "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct",
        "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
    ]

    if model_path not in model_path_list:
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # 可以更大
            do_sample=False,      # 启用采样
            temperature=1.0,     # 增加多样性
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    else:
        outputs = model.generate(
            inputs,
            max_new_tokens=2048,  # 可以更大
            do_sample=False,      # 启用采样
            temperature=1.0,     # 增加多样性
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return outputs

# 预防同名函数
def filter_samenames(i ,fun_string, api_names):
    if lib_name == "torch":
        if fun_string in torch_samename_list:
            if api_names[i+3] == fun_string:
                function_name = fun_string + "_" + str(4)
            elif api_names[i+2] == fun_string:
                function_name = fun_string + "_" + str(3)
            elif api_names[i+1] == fun_string:
                function_name = fun_string + "_" + str(1)
            else:
                function_name = fun_string+ "_" + str(2)
        else:
            function_name = fun_string
    else:
        function_name = fun_string
    return function_name


def get_all_combinations_from_json(api_name, j):
    # path = f'C:/Users/86184/Desktop/torch_combinations.json'
    try:
    # 读取JSON文件
        with open(f'/nasdata/haoyahui/Arg_combinations/{lib_name}_combinations_{j}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取api_name项
        args_combinations = data.get(api_name)
        
    except KeyError:
        return False

    if args_combinations == None:
        k = 1 + j
        with open(f'/nasdata/haoyahui/Arg_combinations/{lib_name}_combinations_{k}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取api_name项
        args_combinations = data.get(api_name)
        return args_combinations, k
    else:
        return args_combinations, j

# 过滤错误组合时断点续生成
def extract_invalid_parameter_combinations():
    #file_path = r'C:\Users\86184\Desktop\test.txt'
    file_path = f'/tmp/Momo_test/error_combinations/{lib_name}_log.txt'
    pattern = r"tf\.keras\.optimizers\.Ftrl 的参数组合 (.*?) 可能不合法"

    result = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 使用finditer查找所有匹配项
            for match in re.finditer(pattern, content, re.DOTALL):
                # 提取参数组合部分
                params_str = match.group(1)
                array = eval(params_str)
                result.append(array)

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
    
    return result