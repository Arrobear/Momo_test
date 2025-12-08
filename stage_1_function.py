from config import *
from generate_prompt import *
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
    if "DeepSeek-R1-Distill-Qwen-32B" in model_path:
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


def extract_clean_json(text: str):
    """
    从大模型输出中抽取 </think> 后的 JSON，
    自动补大括号、去除重复字段、修复常见错误，返回最终解析出的 dict。
    """

    end_tag = "</think>"
    if end_tag not in text:
        return None

    # 1. 获取 </think> 后的内容
    after = text.split(end_tag, 1)[1].strip()

    # 2. 定位 JSON 开始位置
    start = after.find("{")
    if start == -1:
        return None

    fragment = after[start:]

    # 3. 使用大括号平衡提取完整 JSON 字符串
    json_str = balance_json_braces(fragment)

    # 4. 强制去掉末尾非 JSON 内容
    json_str = trim_after_last_brace(json_str)

    # 5. 尝试解析 JSON
    try:
        data = json.loads(json_str)
    except Exception:
        # 如果解析失败，尝试增强修复
        fixed = force_fix_json(json_str)
        try:
            data = json.loads(fixed)
        except Exception:
            return None

    # 6. constraints 去重（如果存在）
    if isinstance(data, dict) and "constraints" in data:
        data["constraints"] = list(dict.fromkeys(data["constraints"]))

    return data


def balance_json_braces(fragment: str) -> str:
    """
    使用大括号平衡算法提取最早闭合的 JSON。
    如果缺失 '}' 则自动补齐。
    """
    balance = 0
    end_index = -1

    for i, ch in enumerate(fragment):
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1

        # 找到完整平衡点
        if balance == 0 and i > 0:
            end_index = i
            break

    # 如果没闭合 → 自动补齐缺失括号
    if end_index == -1:
        return fragment + "}" * balance
    else:
        return fragment[:end_index + 1]


def trim_after_last_brace(s: str) -> str:
    """
    去掉 JSON 后的多余文本，只保留到最后一个大括号。
    """
    last = s.rfind("}")
    if last != -1:
        return s[:last + 1]
    return s


def force_fix_json(s: str) -> str:
    """
    强制修复 JSON：用于 json.loads() 初次失败的情况。
    目前主要操作：
    - 去掉 JSON 后多余部分
    - 补齐缺失括号
    """
    s = trim_after_last_brace(s)

    # 简单检查大括号平衡，如果不够补齐
    open_count = s.count("{")
    close_count = s.count("}")

    if close_count < open_count:
        s += "}" * (open_count - close_count)

    return s



# 封装不同模型的输入输出模式 
def generate_input(prompt, tokenizer, model):

    # model_path_list = [
    #     "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct",
    #     "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
    #     "/home/chaoni/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
    # ]

    # if model_path not in model_path_list:
    
    #     inputs = tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         truncation=True,
    #         max_length=4096,
    #         padding=True
    #     )
    # else:
    # print(111111111111111111)
    inputs = tokenizer.apply_chat_template(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        #padding = "max_length"
        padding=True
    )
    return inputs

def generate_output(inputs, model, tokenizer):
    # model_path_list = [
    #     "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct",
    #     "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B"
    # ]

    # if model_path not in model_path_list:
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=2048,  # 可以更大
    #         do_sample=False,      # 启用采样
    #         temperature=1.0,     # 增加多样性
    #         top_p=1.0,
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=tokenizer.pad_token_id
    #     )
    # else:
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
    k = j
    while True:
        try:
        # 读取JSON文件
            with open(f'/nasdata/haoyahui/Arg_combinations/{lib_name}_combinations_{k}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取api_name项
            args_combinations = data.get(api_name)
            
        except KeyError:
            return False

        if args_combinations == None:
            k += 1
            continue
        else:
            return args_combinations, k

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


#-------------------------------------
# 统一读取json接口
#-------------------------------------
def read_json_api(api_name, file_path, read_mode):
    if read_mode == "combination":
        j = 0
        path = file_path+f'{lib_name}_combinations_{j}.json'
        while True:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if api_name in data:
                return data[api_name]  # 二维数组
            else:
                j += 1
                new_path = file_path+f'{lib_name}_combinations_{j}.json'
                with open(new_path, "r", encoding="utf-8") as f:
                    new_data = json.load(f)
                return new_data[api_name]
            if j > 20:
                break

    elif read_mode == "error_combinations":
        path = file_path+f'error_{lib_name}_combinations.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name] 

    elif read_mode == "arg_space":
        path = file_path+f'{lib_name}_arg_space.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name]

    elif read_mode == "src_code":
        path = file_path+f'{lib_name}_api_sources.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name] 
    elif read_mode == "conditions":
        path = file_path+f'{lib_name}_conditions.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name] 
    else:
        return None

# =========================================
# 保存 API 输入信息的工具函数
# =========================================
def save_api_inputs(api_name, api_inputs, save_path):
    """
    将 {api_name: api_inputs} 增量写入 JSON 文件。
    如果文件不存在则创建，存在则在原内容上追加。
    """
    # 1️⃣ 如果文件不存在 → 创建目录 & 空文件
    if not os.path.exists(save_path):
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
        print(f"[📁 Created] 新文件已创建: {save_path}")

    # 2️⃣ 读取已有数据
    with open(save_path, "r", encoding="utf-8") as f:
        try:
            all_data = json.load(f)
        except json.JSONDecodeError:
            all_data = {}

    # 3️⃣ 合并（增量保存）
    if api_name in all_data:
        all_data[api_name].extend(api_inputs)
    else:
        all_data[api_name] = api_inputs

    # 4️⃣ 写回文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

# =========================================
# 根据规范化的api边界生成测试输入的管道
# =========================================

# 生成复杂参数
def generate_complex_param(api_name, param_name, param_info, constraints, model, tokenizer):
    """
    使用 LLM 生成复杂对象
    """
    api_doc = get_doc(api_name)
    prompt = generate_prompt_4(api_name, param_name, param_info, constraints, api_doc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
    inputs = generate_input(prompt, tokenizer, model)

    # 把inputs放到模型参数所在设备
    inputs = inputs.to(next(model.parameters()).device)

    outputs = generate_output(inputs, model, tokenizer)
    # 解码输出
    outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    complex_input = handle_output(outputs_text, model_path)

    return complex_input["test_values"]

def generate_tensor_param_cases(param_name, param_info, max_dim_limit=256):
    """
    为 Tensor 类型参数生成一系列覆盖性测试样本字符串。
    ✅ 特性：
      - 自动生成 min / mid / max 的维度组合
      - 支持 float、int、bool、complex、bfloat16 等 dtype
      - 自动防止超大形状
      - 统一 dtype 解析，不再生成 torch.torch.float128 之类的错误
    """
    shape_min = param_info.get("shape_min", [1])
    shape_max = param_info.get("shape_max", [3])
    dtypes = param_info.get("dtypes", ["float32"])

    # -------------------------------
    # 1️⃣ shape 边界组合：取 min / mid / max
    # -------------------------------
    shape_cases = []
    for lo, hi in zip(shape_min, shape_max):
        lo = min(int(lo), max_dim_limit)
        hi = min(int(hi), max_dim_limit)

        if lo == hi:
            shape_cases.append([lo])
        else:
            mid = (lo + hi) // 2
            mid = min(mid, max_dim_limit)
            shape_cases.append([lo, mid, hi])

    # 所有组合
    shape_combos = list(itertools.product(*shape_cases))

    # -------------------------------
    # 2️⃣ dtype × shape 组合生成字符串
    # -------------------------------
    samples = []
    for dtype_str in dtypes:
        # 清理 dtype 名称（可能是 "torch.float32"）
        clean_dtype = dtype_str.split(".")[-1].strip()
        if not hasattr(torch, clean_dtype):
            # 避免出现 float128 / 伪类型
            clean_dtype = "float32"

        for shape in shape_combos:
            shape_str = ", ".join(str(s) for s in shape)

            # 根据 dtype 构造表达式
            if any(k in clean_dtype for k in ["float", "half", "bfloat"]):
                expr = f"{param_name} = torch.randn(({shape_str},), dtype=torch.{clean_dtype})"

            elif any(k in clean_dtype for k in ["int", "long"]):
                expr = f"{param_name} = torch.randint(0, 10, ({shape_str},), dtype=torch.{clean_dtype})"

            elif "uint8" in clean_dtype:
                expr = f"{param_name} = torch.randint(0, 256, ({shape_str},), dtype=torch.uint8)"

            elif "bool" in clean_dtype:
                expr = f"{param_name} = (torch.rand(({shape_str},)) > 0.5).to(dtype=torch.bool)"

            elif "complex" in clean_dtype:
                # 对 complex 类型，底层实部 dtype 映射
                base = "float32" if clean_dtype == "complex64" else "float64"
                expr = (
                    f"{param_name} = (torch.randn(({shape_str},), dtype=torch.{base}) + "
                    f"1j * torch.randn(({shape_str},), dtype=torch.{base})).to(dtype=torch.{clean_dtype})"
                )

            else:
                expr = f"# Unsupported dtype: {clean_dtype}"

            samples.append(expr)

    return samples

def generate_scalar_param_cases(param_name, param_info):
    p_type = param_info.get("type")
    lo = param_info.get("min", 0)
    hi = param_info.get("max", 10)

    # 1️⃣ 计算中间点
    mid = (lo + hi) / 2

    # 2️⃣ 生成基本取值
    if p_type == "int":
        # 确保整数范围内不重复
        values = sorted(set([lo, int(mid), hi]))
        samples = [f"{param_name}={v}" for v in values]
    elif p_type == "float":
        # 包括最小、最大、中间、边界偏移
        mid_lo = lo + (mid - lo) / 2
        mid_hi = mid + (hi - mid) / 2
        values = [lo, mid_lo, mid, mid_hi, hi]
        samples = [f"{param_name}={round(v, 6)}" for v in values]
    else:
        raise ValueError(f"Unsupported type: {p_type}")

    return samples

# 生成简单参数
def generate_sample_param(api_name, param, param_info):
    """
    根据 param_info 的类型生成单个参数样本。
    支持 Tensor、int、float、bool、str、optional、choices 等。
    """
    p_type = param_info.get("type")

    # 1️⃣ Tensor 类型
    if "Tensor" in p_type:

        return generate_tensor_param_cases(param, param_info)

    # 2️⃣ 数值型参数
    elif p_type in ["Int", "Float"]:
        return generate_scalar_param_cases(param, param_info)

    # 3️⃣布尔型参数
    elif "Bool" in p_type:
        samples = [f"{param}=True", f"{param}=False"]
        return samples

    # 4️⃣ 字符串参数（无 choices）
    elif "Str" in p_type and "choices" not in param_info:
        length = param_info.get("length", 5)
        samples = []
        for _ in range(2):  # 生成两个不同字符串
            s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
            samples.append(f"{param}={s}")
        return samples

    # 5️⃣ 可选参数（可能为 None）
    elif "Optional" in p_type:
        samples = []
        if "choices" in param_info:
            # 包含 None + 所有枚举选项
            samples = [f"{param}=None"] + [f"{param}={choice}" for choice in param_info["choices"]]
        else:
            # 默认包含 None + 一个示例值
            samples = [f"{param}=None", f"{param}=some_value"]
        return samples

    # 6️⃣ 有 choices（枚举）参数
    elif "choices" in param_info:
        choices = param_info["choices"]
        samples = [f"{param}=None"] + [f"{param}={choice}" for choice in choices] 
        return samples
    else:
        raise ValueError(f"[{api_name}] Unsupported type: {p_type}")

# 检查约束条件
def check_constraints(combo, constraints):


    return False

# 将元组列表转换为字典列表
def convert_list_to_dict_list(data_list):
    """
    data_list 是形如:
        [('input = ...', 'dim=1', 'index = ...'), ...]
    返回:
        [{'input': '...', 'dim': '1', 'index': '...'}, ...]
    """
    import ast

    def parse_assignment(expr: str):
        """
        将字符串 'key = value' 解析成字典 {key: value}
        value 保留为原始表达式字符串，不执行 eval
        """
        if "=" not in expr:
            raise ValueError("表达式必须包含 '='")

        key, value = expr.split("=", 1)
        key = key.strip()
        value = value.strip()

        return {key: value}

    result = []
    for tup in data_list:
        item_dict = {}
        for expr in tup:
            # 使用 parse_assignment 解析表达式
            parsed = parse_assignment(expr)
            item_dict.update(parsed)
        result.append(item_dict)
    
    return result



def generate_test_inputs_from_api_boundaries(api_name, api_boundaries, model=None, tokenizer=None):
    """
    根据 API 的边界规范，生成满足约束的测试输入组合。
    """
    params = api_boundaries.get("params", {})
    constraints = api_boundaries.get("constraints", [])

    # 1️⃣ 为每个参数生成候选样本
    candidate_dict = {}
    for param_name, param_info in params.items():
        p_type = param_info.get("type")
        if p_type in ["Tensor", "int", "float", "bool", "str", "optional"]:
            candidate_dict[param_name] = generate_sample_param(api_name, param_name, param_info)
        else:
            # 使用模型生成复杂参数
            candidate_dict[param_name] = [generate_complex_param(api_name, param_name, param_info, constraints, model, tokenizer)]

    # 2️⃣ 生成所有参数的笛卡尔积组合
    keys = list(candidate_dict.keys())
    all_combos = list(itertools.product(*[candidate_dict[k] for k in keys]))

    # 3️⃣ 约束筛选
    valid_inputs = []
    i = 1
    length = len(all_combos)
    for combo in all_combos:
        print("第"+str(i)+"/"+str(length)+"个")
        i += 1
        if check_constraints(combo, constraints):

            valid_inputs.append(combo)

    # 4️⃣ 转换为字典列表
    new_combos = convert_list_to_dict_list(valid_inputs)

    return new_combos




# test_bundary = {
# "params": {
# "input": {
# "type": "Tensor",
# "shape_min": [1, 1, 1],
# "shape_max": [128, 4096, 65536],
# "dtypes": ["torch.float16", "torch.bfloat16", "torch.float32", "torch.float64", "torch.complex64", "torch.complex128"]
# },
# "dim": {
# "type": "int",
# "min": 0,
# "max": 3
# },
# "index": {
# "type": "Tensor",
# "shape_min": [1],
# "shape_max": [4096],
# "dtypes": ["torch.int32", "torch.int64"]
# }
# },
# "constraints": [
# "input.dtype == index.dtype",
# "input.dim() >= 1",
# "index.shape[0] >= 1",
# "dim >= 0 and dim < input.dim()",
# "index.shape[0] == input.shape[dim]"
# ]
# }
test_bundary = {
"params": {
"input": {
"type": "Tensor",
"shape_min": [1, 1, 1],
"shape_max": [128, 4096, 65536],
"dtypes": ["torch.float16", "torch.bfloat16", "torch.float32", "torch.float64", "torch.complex64", "torch.complex128"]
},
"dim": {
"type": "int",
"min": 0,
"max": 3
},
"index": {
"type": "Tensor",
"shape_min": [1],
"shape_max": [4096],
"dtypes": ["torch.int32", "torch.int64"]
}
},
"constraints": [
"input.dtype == index.dtype",
"input.dim() >= 1",
"index.shape[0] >= 1",
"dim >= 0 and dim < input.dim()",
"index.shape[0] == input.shape[dim]"
]
}

# a = generate_test_inputs_from_api_boundaries(api_name = "1", api_boundaries = test_bundary, model=None, tokenizer=None)
# for i in a:
#     print(i)



def convert_input_to_string(params):
    """Convert all Tensors in params to torch.randn string expressions."""
    stringified = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            shape = tuple(v.shape)
            dtype = str(v.dtype)
            # 简化表达：float32 → 默认 torch.randn
            if dtype == "torch.float32":
                stringified[k] = f"torch.randn{shape}"
            else:
                stringified[k] = f"torch.randn{shape}, dtype={dtype}"
        else:
            stringified[k] = v
    return stringified

def execute_api_template(run_api_func, test_inputs, log_path="error_log.json",
                         timeout_s=30, perf_time_threshold=5.0, mem_threshold_gb=8):
    """
    执行 run_api 函数，对输入进行批量测试。
    仅记录出错样例（Crash / Numerical / Performance）。
    """

    results = {
        "crash": [],
        "numerical": [],
        "performance": []
    }

    def record_issue(issue_type, input_data, err_msg):
        # 转换输入为字符串表达
        safe_input = convert_input_to_string(input_data)
        results[issue_type].append({
            "input": safe_input,
            "error": err_msg
        })

    def get_memory_usage_gb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    for i, params in enumerate(test_inputs):
        torch.cuda.empty_cache()
        gc.collect()
        start_mem = get_memory_usage_gb()
        start_time = time.time()

        try:
            # 执行 API
            result = run_api_func(**params)
            elapsed = time.time() - start_time
            end_mem = get_memory_usage_gb()

            # 性能异常
            if elapsed > perf_time_threshold or (end_mem - start_mem) > mem_threshold_gb:
                record_issue("performance", params,
                             f"Runtime {elapsed:.2f}s, MemDelta {end_mem - start_mem:.2f} GB")

            # 数值异常
            def has_nan_or_inf(t):
                return isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any())

            if isinstance(result, torch.Tensor):
                if has_nan_or_inf(result):
                    record_issue("numerical", params, "NaN or Inf in output")
            elif isinstance(result, (tuple, list)):
                for r in result:
                    if has_nan_or_inf(r):
                        record_issue("numerical", params, "NaN or Inf in tuple output")
                        break

        except RuntimeError as e:
            err_msg = str(e)
            if "CUDA" in err_msg or "device-side assert" in err_msg or "out of memory" in err_msg:
                record_issue("crash", params, f"CUDA-related crash: {err_msg}")
            else:
                record_issue("crash", params, f"RuntimeError: {err_msg}")

        except KeyboardInterrupt:
            print("⛔️ Interrupted by user.")
            break

        except Exception:
            record_issue("crash", params, traceback.format_exc())

        # 超时检测
        elapsed = time.time() - start_time
        if elapsed > timeout_s:
            record_issue("performance", params, f"Timeout: exceeded {timeout_s}s")

    # 保存日志（仅包含报错项）
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n⚠️  Error log written to {log_path}")
    for k, v in results.items():
        print(f"  {k.upper():12s}: {len(v)} cases")

    return results