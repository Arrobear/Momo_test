from config import *
from generate_prompt import *
from generate_input import *
import time

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

call_llm_with_retry(client, model, messages, **kwargs)：带无限重试的LLM API调用
'''


def call_llm_with_retry(client, model, messages, **kwargs):
    kwargs.setdefault("timeout", 300)
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content, retrying...")
            return content
        except Exception as e:
            print(f"[API 错误] {e}，3秒后重试...")
            time.sleep(3)



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
def get_doc(function_name: str) -> str:
    """
    根据完整的API名称（如 'torch.bitwise_not' 或 'scipy.optimize.minimize'）
    动态导入相关模块并获取其文档字符串。如果该API是类内方法且自身无文档，
    将自动回退获取其所属类的文档字符串。
    
    参数:
        function_name (str): 完整的API调用路径字符串。
        
    返回:
        str: 提取到的文档字符串。如果未找到或无文档，返回相应的提示信息。
    """
    if not function_name or not isinstance(function_name, str):
        return "错误：输入必须是非空的字符串。"

    if USE_SOURCE_RESOLVER:
        from source_resolver import get_docstring_from_source
        doc = get_docstring_from_source(function_name, lib_gitname)
        if doc:
            return doc
        return None

    parts = function_name.split('.')
    
    # 采用“降级导入”策略：从最长的路径开始尝试导入模块
    for i in range(len(parts), 0, -1):
        module_name = '.'.join(parts[:i])
        try:
            # 尝试动态导入模块
            obj = importlib.import_module(module_name)
            
            # 【核心修改1】：引入 parent_obj 用于追踪目标对象的上级节点
            parent_obj = None
            
            # 如果模块导入成功，依次向下获取具体的属性（类、函数、方法等）
            for attr in parts[i:]:
                parent_obj = obj
                obj = getattr(obj, attr)
            
            # 优先使用 inspect.getdoc() 获取清理过缩进的文档字符串
            doc = inspect.getdoc(obj)
            if not doc and hasattr(obj, '__doc__'):
                doc = obj.__doc__
            
            # 【核心修改2】：针对无文档的 API，如果是类内方法，获取其父类文档
            if not doc and parent_obj is not None and inspect.isclass(parent_obj):
                doc = inspect.getdoc(parent_obj)
                if not doc and hasattr(parent_obj, '__doc__'):
                    doc = parent_obj.__doc__
                # if doc:
                #     print(f"提示：API '{function_name}' 自身无文档，已回退提取其所属类 '{parent_obj.__name__}' 的文档。")

            if doc:
                return doc
            else:
                # print(f"提示：找到了API '{function_name}'，但该API及其所属类（若有）均没有编写文档字符串。")
                return None
            
        except (ImportError, AttributeError):
            # 如果当前层级导入失败或找不到属性，继续缩短模块路径尝试
            continue
        except Exception as e:
            # 捕获库初始化时可能抛出的其他运行时异常
            print(f"错误：在提取 '{function_name}' 时发生异常: {str(e)}")
            return None
            
    print(f"错误：无法找到API '{function_name}'。请确保输入的API名称正确，并且环境中已安装对应的第三方库。")        
    return None

# 清理无文档的 API：从源文件中物理删除无文档的 API 条目（保留有效 API 的原始签名）。
def clean_undocumented_apis_from_file(file_path):
    """
    使用现有的 read_file 提取 API 名并验证文档，
    将无文档的 API 从源文件中物理删除（同时保留有效 API 的原始签名）。
    """
    if not os.path.exists(file_path):
        print(f"[-] 错误：找不到文件 {file_path}")
        return

    # 1. 使用你已有的 read_file 获取干净的 API 名列表
    # (假设 read_file 已经在当前命名空间或被正确导入)
    api_names = read_file(file_path)
    
    valid_api_names = set()
    removed_count = 0

    print(f"[*] 开始验证 {lib_name} 的 API 文档，共 {len(api_names)} 个待测条目...")

    # 2. 核心逻辑：验证每个 API 是否有文档
    for api_name in api_names:
        doc = get_doc(api_name)
        if doc is None:
            print(f"[-] API '{api_name}' 没有找到文档，准备移除。")
            removed_count += 1
        else:
            valid_api_names.add(api_name)

    # 3. 如果有需要移除的 API，读取原文件并进行安全覆写
    if removed_count > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
            
        valid_lines_to_write = []
        for line in original_lines:
            original_str = line.strip()
            if not original_str:
                continue
            
            # 获取当前行的纯净名称用于比对字典（兼容带有签名的源文件行）
            current_clean_name = original_str.split('(')[0]
            
            # 如果该 API 在有效集合中，则保留其包含参数签名的整行
            if current_clean_name in valid_api_names:
                valid_lines_to_write.append(original_str)

        # 覆写回源文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for valid_line in valid_lines_to_write:
                f.write(f"{valid_line}\n")
                
        print(f"[*] 清理完成！已从源文件中物理删除 {removed_count} 个无文档的 API。当前剩余有效 API: {len(valid_lines_to_write)} 个。")
    else:
        print(f"[*] 验证完成！所有 API 均拥有文档，源文件未作修改。")


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
    json_filename = path = root_path + f'/documentation/conditions/{lib_name}_conditions.json'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "conditions", json_filename)
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if api_name not in data:
        print(f"[提示] API '{api_name}' 未在 conditions 文件中找到，跳过")
        return []
    
    if "Parameter type" not in data[api_name]:
        return []
    
    # 获取所有 key
    return list(data[api_name]["Parameter type"].keys())
    
    # 过滤掉 "self" (忽略大小写可以使用 p.lower() != 'self')
    # return [p for p in params if p != "self"]
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
    # 1. 预处理 new_data：确保其转化为字典
    if not new_data:
        condition_dict = {}
    elif isinstance(new_data, dict):
        condition_dict = new_data
    else:
        try:
            # 解析字符串为 Python 字典
            condition_dict = json.loads(new_data)
        except (json.JSONDecodeError, TypeError):
            print(f"[错误] 无法解析 {fun_string} 的数据格式")
            return

    # 2. 确保目标文件夹存在
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 3. 读取原始数据
    data = {}
    if os.path.exists(path):
        try:
            # 检查文件大小，避免读取空文件导致的 JSONDecodeError
            if os.path.getsize(path) > 0:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        except json.JSONDecodeError:
            # 如果文件损坏，备份并初始化（科研严谨性：防止覆盖已有数据）
            print(f"[警告] {path} 文件损坏，已初始化空字典")
            data = {}

    # 4. 更新数据
    data[fun_string] = condition_dict

    # 5. 写回文件
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[错误] 写入文件失败: {e}")

# 获取JSON文件中的api_conditions
def get_api_conditions(fun_string, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 直接获取指定函数名对应的条件字典
        return data.get(fun_string, None)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        # add_log(f"Error reading file: {e}")
        print(f"Error reading file: {e}")
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
    if model_path == "/nasdata/Model/Meta-Llama-3-70B-Instruct":
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
            print("[-] 错误：输出中未找到 '</think>' 标签，无法提取 JSON 内容。")
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

# 从大模型输出中抽取 <tool_call> 后的 JSON
def extract_clean_list(outputs_text: str) -> list:
    """
    通过静态 AST 解析大模型输出的复杂列表文本，转换为严格可 JSON 序列化的 Python 列表。
    保留了结构，所有非常规对象均被降级为字符串表示。
    """
    text = outputs_text.strip()
    if not text:
        return []

    try:
        # mode='eval' 表示期望解析一个表达式树，该过程完全静态
        tree = ast.parse(text, mode='eval')
    except SyntaxError as e:
        # 大模型输出存在根本性语法错误时的阻断
        return [f"SyntaxError_during_parsing: {e}"]

    # 校验根节点类型
    if not isinstance(tree.body, ast.List):
        return [f"TypeError: Expected list, got {type(tree.body).__name__}"]

    def to_json_safe(node):
        """递归解析 AST 节点并映射为 JSON 安全类型"""
        # 1. 处理基础常量
        if isinstance(node, ast.Constant):
            val = node.value
            # JSON 原生支持的数据类型
            if isinstance(val, (str, int, float, bool, type(None))):
                # 拦截 JSON 标准不完全支持的特殊浮点数
                if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
                    return str(val)
                return val
            # bytes, complex 等非 JSON 类型转为字面量字符串
            return repr(val)
        
        # 2. 处理列表和元组 (统一降级为 JSON Array)
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [to_json_safe(elt) for elt in node.elts]
        
        # 3. 处理字典 (JSON Object)
        elif isinstance(node, ast.Dict):
            safe_dict = {}
            for k, v in zip(node.keys, node.values):
                # JSON 的键强制要求为字符串
                safe_key = str(to_json_safe(k)) if k is not None else "null"
                safe_dict[safe_key] = to_json_safe(v)
            return safe_dict
        
        # 4. 其他所有复杂节点 (如 object(), T.__('x'), lambda)
        # 使用 ast.unparse 将其重新生成为标准文本
        else:
            try:
                return ast.unparse(node)
            except Exception:
                return "<Unparseable_AST_Node>"

    # 遍历外层列表元素
    return [to_json_safe(elt) for elt in tree.body.elts]



def extract_clean_json(text: str):
    # 1. 定位有效区域
    end_tag = "</think>"
    if end_tag in text:
        after = text.split(end_tag, 1)[1].strip()
    else:
        after = text.strip()

    # 2. 提取 JSON 片段
    start = after.find("{")
    if start == -1:
        return None
    
    # 查找最后一个大括号，粗略截取
    end = after.rfind("}")
    if end == -1 or end < start:
        return None
    
    json_str = after[start:end+1]

    # 3. 预处理 Python 关键字（仅在不在引号内时替换，防止误杀）
    def replace_keep_quotes(m):
        s = m.group(0)
        if s.startswith('"') or s.startswith("'"):
            return s
        s = re.sub(r'\bNone\b', 'null', s)
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        return s

    # 简单通过正则区分字符串内外进行替换
    json_str = re.sub(r'("[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'|[^"\']+)', replace_keep_quotes, json_str)

    # 4. 解析逻辑
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # 调用外部修复函数（可选依赖，失败则跳过）
        try:
            from json_repair import repair_json
            json_str = repair_json(json_str)
            data = json.loads(json_str)
        except Exception:
            # 最后的保底尝试：基础符号修复
            try:
                # 如果没有 balance_json_braces，可在此实现简单的大括号对齐
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += '}' * (open_braces - close_braces)
                data = json.loads(json_str)
            except:
                return None

    # 5. 后处理
    if isinstance(data, dict) and "constraints" in data:
        if isinstance(data["constraints"], list):
            data["constraints"] = list(dict.fromkeys(data["constraints"]))

    return data
# 使用大括号平衡算法提取最早闭合的 JSON。
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

# 去掉 JSON 后的多余文本，只保留到最后一个大括号。
def trim_after_last_brace(s: str) -> str:
    """
    去掉 JSON 后的多余文本，只保留到最后一个大括号。
    """
    last = s.rfind("}")
    if last != -1:
        return s[:last + 1]
    return s

# 强制修复 JSON：用于 json.loads() 初次失败的情况。
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
            with open(root_path + f'/documentation/arg_combinations/{lib_name}_combinations_{k}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取api_name项
            
            args_combinations = data.get(api_name)

        except (KeyError, FileNotFoundError):
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
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        if api_name in data:
            return data[api_name]
        else:
            return None

    elif read_mode == "error_combinations":
        path = file_path+f'error_{lib_name}_combinations.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name] 

    elif read_mode == "arg_space":
        j = 0
        path = file_path+f'{lib_name}_arg_space_{j}.json'
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        if api_name in data:
            return data[api_name]
        else:
            return None
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
    elif read_mode == "boundary":
        path = file_path+f'cut_{lib_name}_boundary_0.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name] 
    elif read_mode == "default_input":
        path = file_path+f'{lib_name}_default_inputs_0.json'
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if api_name in data:
            return data[api_name] 
    elif read_mode == "inputs":
        j = 0
        while j <= 20:
            path = file_path + f'{lib_name}_inputs_{j}.json'
            if not os.path.exists(path):
                break
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if api_name in data:
                    return data[api_name]
            except (json.JSONDecodeError, IOError):
                pass
            j += 1
        return None
    elif read_mode == "case":
        j = 0
        while j <= 20:
            path = file_path + f'{lib_name}_case_{j}.json'
            if not os.path.exists(path):
                break
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if api_name in data:
                    return data[api_name]
            except (json.JSONDecodeError, IOError):
                pass
            j += 1
        return None
    elif read_mode == "cut_combination":
        j = 0
        path = file_path+f'{lib_name}_cut_combinations_{j}.json'
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        if api_name in data:
            return data[api_name]
        else:
            return None
    else:
        return None

# =========================================
# 保存 API 输入信息的工具函数
# =========================================
def _deep_merge(existing, new):
    """递归增量合并：dict 递归合并，list 去重追加，其他类型用新值覆盖"""
    if isinstance(existing, dict) and isinstance(new, dict):
        for k, v in new.items():
            if k in existing:
                existing[k] = _deep_merge(existing[k], v)
            else:
                existing[k] = v
        return existing
    elif isinstance(existing, list) and isinstance(new, list):
        for item in new:
            if item not in existing:
                existing.append(item)
        return existing
    else:
        return new


def save_api_inputs(api_name, api_inputs, save_path):
    """
    将 {api_name: api_inputs} 增量写入 JSON 文件。
    如果文件不存在则创建，存在则在原内容上增量合并，不会覆盖已有数据。
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

    # 3️⃣ 增量合并：已有 key → 深度合并，新 key → 直接添加
    if api_name in all_data:
        all_data[api_name] = _deep_merge(all_data[api_name], api_inputs)
    else:
        all_data[api_name] = api_inputs

    # 4️⃣ 清理代理字符后写回文件
    all_data = _clean_surrogates(all_data)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


def _clean_surrogates(obj):
    """递归清除数据中的孤立代理字符 (U+D800~U+DFFF)，这些字符无法被 UTF-8 编码。"""
    if isinstance(obj, str):
        return ''.join(
            c if ord(c) < 0xD800 or ord(c) > 0xDFFF else '�'
            for c in obj
        )
    elif isinstance(obj, dict):
        return {k: _clean_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_surrogates(v) for v in obj]
    return obj


# =========================================
# 根据规范化的api边界生成测试输入的管道
# =========================================

# 生成复杂参数
def generate_complex_param(api_name, param_name, param_info, constraints, model, tokenizer):
    """
    使用 LLM 生成复杂对象
    """
    api_doc = get_doc(api_name)
    prompt = generate_prompt_5(api_name, param_name, param_info, constraints, api_doc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
    inputs = generate_input(prompt, tokenizer, model)

    # 把inputs放到模型参数所在设备
    inputs = inputs.to(next(model.parameters()).device)

    outputs = generate_output(inputs, model, tokenizer)
    # 解码输出
    outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    complex_input = extract_clean_json(outputs_text, model_path)
    complex_input = json.loads(complex_input)
    complex_input_list = complex_input["test_values"]
    samples = [f"{param_name}={v}" for v in complex_input_list]
    return samples


# -------------------------------------------------------
# 约束检查
# -------------------------------------------------------
# -------------------------------------------------------
#check_constraints()
# -------------------------------------------------------
def check_constraints(combo_dict, constraints, default_inputs):
    """
    检查当前的参数组合是否满足所有约束条件。
    遇到不合法的约束语句，默认视为满足 (True)。
    """
    if not constraints:
        return True
        
    merged_inputs = {**default_inputs, **combo_dict}
    
    env = {}
    for k, v in merged_inputs.items():
        if isinstance(v, str):
            if v == "null" or v == "None":
                env[k] = None
            elif v == "True":
                env[k] = True
            elif v == "False":
                env[k] = False
            else:
                try:
                    _eval_ns = {"torch": torch} if "torch" in sys.modules else {}
                    env[k] = eval(v, _eval_ns)
                except Exception:
                    env[k] = v
        else:
            env[k] = v
            
    # 3. 逐一验证约束条件
    for constraint in constraints:
        try:
            # 尝试执行约束判断表达式
            _eval_ns = {"torch": torch} if "torch" in sys.modules else {}
            result = eval(constraint, _eval_ns, env)
            # 只有当语句合法且明确返回 False 时，才判定为不满足
            if not result:
                return False
        except Exception as e:
            # 【核心修改点】
            # 如果 constraint 不是合法的 Python 语句（如 SyntaxError）
            # 或者生成的测试值缺少对应属性（如对整数 54 取 .ndim 引发 AttributeError）
            # 按照你的需求，这里捕获异常并放行（默认为 True），继续检查下一个约束
            # print(f"警告: 约束 '{constraint}' 无法评估，跳过。原因: {e}")
            continue
            
    return True


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



def generate_test_inputs_from_api_boundaries(api_name, api_boundaries, model=None, tokenizer=None, default_inputs=None):
    """
    根据 API 的边界规范，生成满足约束的测试输入组合。
    """
    params = api_boundaries.get("params", {})
    constraints = api_boundaries.get("constraints", [])
    if default_inputs is None:
        default_inputs = {}

    # 1️⃣ 为每个参数生成候选样本
    candidate_dict = {}
    for param_name, param_info in params.items():
        param_input = generate_sample_param(api_name, param_name, param_info)
        if param_input == "complex":
            # 使用模型生成复杂参数
            candidate_dict[param_name] = generate_complex_param(api_name, param_name, param_info, constraints, model, tokenizer)
        else:
            candidate_dict[param_name] = param_input

    # 2️⃣ 生成所有参数的笛卡尔积组合
    keys = list(candidate_dict.keys())
    all_combos_tuples = list(itertools.product(*[candidate_dict[k] for k in keys]))

    # 3️⃣ 约束筛选
    valid_inputs = []
    i = 1
    length = len(all_combos_tuples)
    for combo_tuple in all_combos_tuples:
        print(f"第 {i}/{length} 个")
        i += 1
        
        # 【修正】将 tuple 转换为带有参数名的字典
        combo_dict = dict(zip(keys, combo_tuple))
        
        if check_constraints(combo_dict, constraints, default_inputs):
            # 将字典形式加入有效列表
            valid_inputs.append(combo_dict)

    # 4️⃣ 返回经过筛选的有效组合 (无需再去转换 all_combos)
    return valid_inputs




def convert_input_to_string(params):
    """Convert all Tensors in params to torch.randn string expressions."""
    stringified = {}
    for k, v in params.items():
        if "torch" in sys.modules and isinstance(v, torch.Tensor):
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
        if "torch" in sys.modules:
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
                if "torch" in sys.modules:
                    return isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any())
                return False

            if "torch" in sys.modules and isinstance(result, torch.Tensor):
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


def keep_min_max_per_pattern(arg_combinations, error_combinations, filter_params):
    """
    对 filter_params 的存在性模式做覆盖。
    每个模式最多保留2个组合（参数数目最小 & 最大）。
    返回扁平的组合列表（去重）。
    """
    filter_params = list(filter_params)  # 固定顺序
    error_set = {tuple(c) for c in error_combinations or []}

    # 预过滤合法组合
    valid = [c for c in arg_combinations if tuple(c) not in error_set]

    # pattern(tuple[bool]) -> (min_size, min_comb), (max_size, max_comb)
    best = {}

    def pattern_of(cset):
        return tuple(p in cset for p in filter_params)

    for comb in valid:
        cset = set(comb)
        pat = pattern_of(cset)
        size = len(cset)

        if pat not in best:
            best[pat] = {
                "min": (size, comb),
                "max": (size, comb)
            }
        else:
            if size < best[pat]["min"][0]:
                best[pat]["min"] = (size, comb)
            if size > best[pat]["max"][0]:
                best[pat]["max"] = (size, comb)

    # 扁平化 + 去重（保持首次出现顺序）
    result = []
    seen = set()

    for pat in product([False, True], repeat=len(filter_params)):
        if pat not in best:
            continue

        min_comb = best[pat]["min"][1]
        max_comb = best[pat]["max"][1]

        for c in (min_comb, max_comb):
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                result.append(c)

    return result


# 利用arg_space中的onjuncts 对combinations进行剪枝 → cut_combinations

def cut_combinations(api_names):

    # 从 conjuncts 提取依赖参数
    def extract_filter_params(conjuncts, all_param):
        filter_params = set()
        for conjunct in conjuncts:
            for param in all_param:
                if param in conjunct:
                    filter_params.add(param)
        return filter_params

    if lib_name != "torch":
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        j = 0
        path = root_path + f"/documentation/arg_combinations/{lib_name}_cut_combinations_{j}.json"
        length_api_names = len(api_names)
        for i in range(0, length_api_names):
            api_name = filter_samenames(i, api_names[i], api_names)
            condition = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
            # print(condition)
            if not condition:
                # print(11111111111111111111)
                continue
            all_param = []
            for key in condition["Parameter type"]:
                all_param.append(key)
            arg_combinations = read_json_api(api_name=api_name, file_path=f"../documentation/arg_combinations/", read_mode="combination")
            error_combinations = read_json_api(api_name=api_names, file_path=f"../documentation/error_combinations/", read_mode="error_combination")
            arg_spaces = read_json_api(api_name=api_names[i], file_path=f"../documentation/arg_space/", read_mode="arg_space")
            if arg_combinations is None or arg_spaces is None:
                continue
            if error_combinations is None:
                error_combinations = []
        
            cut_combination = []

            for arg_space in arg_spaces:
                space_id = arg_space.get("id")
                conjuncts = arg_space.get("conjuncts", [])

                # 1. 提取该 space 相关的参数
                filter_params = extract_filter_params(conjuncts, all_param)

                space_combinations = keep_min_max_per_pattern(
                    arg_combinations=arg_combinations,
                    error_combinations=error_combinations,
                    filter_params=filter_params
                )

                if space_combinations:
                    cut_combination.append({
                        "id": space_id,
                        "combinations": space_combinations
                    })
                #存储至json
            if is_file_too_large(path, max_size_mb=1000):
                j+=1
                path = root_path + f"/documentation/arg_combinations/{lib_name}_cut_combinations_{j}.json"
                save_api_inputs(api_name, cut_combination, path)
            else:
                save_api_inputs(api_name, cut_combination, path)

            print(f"进度"+str(i+1)+"/"+str(len(api_names)))
            # if i == 0:
            # break


    elif lib_name == "tf":
        pass
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        # prompt = generate_prompt_3(api_names)
        # 将输入存入json文件

    # 添加新的深度学习库
    else:
        pass

    return





def save_and_paginate(api_name, api_run_results, path, root_path, lib_name, j, page_pattern=None):
    if is_file_too_large(path, max_size_mb=1000):
        j += 1
        if page_pattern:
            path = page_pattern.format(root_path=root_path, lib_name=lib_name, j=j)
        else:
            path = root_path + f'/documentation/results/{lib_name}_result_{j}.json'
    # 将符合差分目标的字典列表 api_run_results 写入指定 api_name 键下
    save_api_inputs(api_name, api_run_results, path)
    return j, path


def safe_serialize(obj):
    """
    安全序列化函数，确保 V2 的输出与 V1 记录格式统一，防止对比误报。
    当 repr 触发 RecursionError 时，返回递归 bug 标记。
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    try:
        return repr(obj)
    except RecursionError:
        return f"[RECURSION_BUG] {type(obj).__name__}: repr() 触发了递归深度超限，疑似对象中存在循环引用或自引用结构"

def generate_bug_report(result_path=None, output_path=None):
    """
    从基线 JSON 中提取所有 recursion_bug 条目，生成 bug 汇总报告。

    :param result_path: 基线结果 JSON 路径，默认 results/{lib_name}_v1_baseline.json
    :param output_path: bug 报告输出路径，默认 results/{lib_name}_recursion_bugs.json
    :return: 汇总 dict {api_name: [bug_entries]}
    """
    if result_path is None:
        result_path = root_path + f'/documentation/results/{lib_name}_v1_baseline.json'
    if output_path is None:
        output_path = root_path + f'/documentation/results/{lib_name}_recursion_bugs.json'

    if not os.path.exists(result_path):
        print(f"[bug_report] 基线文件不存在: {result_path}")
        return {}

    with open(result_path, "r", encoding="utf-8") as f:
        try:
            all_data = json.load(f)
        except json.JSONDecodeError:
            print(f"[bug_report] 基线文件 JSON 解析失败: {result_path}")
            return {}

    bug_manifest = {}
    for api_name, cases in all_data.items():
        bug_cases = []
        for idx, case in enumerate(cases):
            result_str = str(case.get("函数返回结果", ""))
            if case.get("函数运行状态") == "recursion_bug" or case.get("bug_category") == "recursion" or result_str.startswith("[RECURSION_BUG]"):
                location = case.get("bug_location")
                if not location:
                    location = "result_repr" if result_str.startswith("[RECURSION_BUG]") else "unknown"
                bug_cases.append({
                    "case_index": idx,
                    "inputs": case.get("测试输入", {}),
                    "bug_location": location,
                    "detail": result_str
                })
        if bug_cases:
            bug_manifest[api_name] = bug_cases

    # 写报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bug_manifest, f, indent=4, ensure_ascii=False)

    # 打印汇总
    total_bugs = sum(len(v) for v in bug_manifest.values())
    print("\n" + "=" * 60)
    print(f"  递归 Bug 汇总: 共 {total_bugs} 个用例, 涉及 {len(bug_manifest)} 个 API")
    print("=" * 60)
    for api_name, bugs in sorted(bug_manifest.items()):
        locations = set(b["bug_location"] for b in bugs)
        print(f"  {api_name}: {len(bugs)} 个用例, 触发位置: {', '.join(sorted(locations))}")
    print(f"\n详细报告已保存至: {output_path}")

    return bug_manifest


def generate_timeout_bug_report(result_path=None, output_path=None):
    """
    从基线 JSON 中提取所有 timeout 条目，生成超时 bug 汇总报告。

    :param result_path: 基线结果 JSON 路径，默认 results/{lib_name}_v1_baseline.json
    :param output_path: bug 报告输出路径，默认 results/{lib_name}_timeout_bugs.json
    :return: 汇总 dict {api_name: [bug_entries]}
    """
    if result_path is None:
        result_path = root_path + f'/documentation/results/{lib_name}_v1_baseline.json'
    if output_path is None:
        output_path = root_path + f'/documentation/results/{lib_name}_timeout_bugs.json'

    if not os.path.exists(result_path):
        print(f"[timeout_bug_report] 基线文件不存在: {result_path}")
        return {}

    # 支持分页加载
    bug_manifest = {}
    base_no_ext = result_path.replace('.json', '')
    for j in range(20):
        if j == 0:
            page_path = result_path
        else:
            page_path = f"{base_no_ext}_{j}.json"
        if not os.path.exists(page_path):
            break
        with open(page_path, "r", encoding="utf-8") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                continue

        for api_name, cases in all_data.items():
            bug_cases = []
            for idx, case in enumerate(cases):
                result_str = str(case.get("函数返回结果", ""))
                if case.get("函数运行状态") == "timeout" or case.get("bug_category") == "timeout" or result_str.startswith("[TIMEOUT]"):
                    location = case.get("bug_location", "api_execution")
                    bug_cases.append({
                        "case_index": idx,
                        "inputs": case.get("测试输入", {}),
                        "bug_location": location,
                        "detail": result_str
                    })
            if bug_cases:
                if api_name not in bug_manifest:
                    bug_manifest[api_name] = []
                bug_manifest[api_name].extend(bug_cases)

    # 写报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bug_manifest, f, indent=4, ensure_ascii=False)

    # 打印汇总
    total_bugs = sum(len(v) for v in bug_manifest.values())
    print("\n" + "=" * 60)
    print(f"  超时 Bug 汇总: 共 {total_bugs} 个用例, 涉及 {len(bug_manifest)} 个 API")
    print("=" * 60)
    for api_name, bugs in sorted(bug_manifest.items()):
        locations = set(b["bug_location"] for b in bugs)
        print(f"  {api_name}: {len(bugs)} 个用例, 触发位置: {', '.join(sorted(locations))}")
    print(f"\n详细报告已保存至: {output_path}")

    return bug_manifest

def _normalize_address(s):
    """将字符串中的内存地址 0x... 替换为 0xXXXX，消除跨进程地址差异"""
    import re
    return re.sub(r'0x[0-9a-fA-F]+', '0xXXXX', s)


def compare_results(v1_result, v2_result, status_v1, status_v2):
    """
    差分断言逻辑。
    比较 V1 和 V2 的状态及输出结果。
    """
    if status_v1 != status_v2:
        return False, f"状态不一致: V1 [{status_v1}] vs V2 [{status_v2}]"

    # 转换为字符串后比对，先规范化内存地址避免跨进程地址差异导致的误报
    s1 = _normalize_address(str(v1_result))
    s2 = _normalize_address(str(v2_result))
    if s1 != s2:
        return False, f"输出不一致:\n  V1: {v1_result}\n  V2: {v2_result}"

    return True, "一致"


def get_function_signature_str(function_name: str) -> str:
    """
    使用 inspect.signature() 从已安装的库中反射获取函数的完整签名。
    仅在 APIdef.txt 不含签名时作为 fallback。
    如果导入失败，回退返回 function_name。
    """
    if not function_name or '.' not in function_name:
        return function_name

    parts = function_name.split('.')
    for i in range(len(parts), 0, -1):
        module_name = '.'.join(parts[:i])
        try:
            obj = importlib.import_module(module_name)
            for attr in parts[i:]:
                obj = getattr(obj, attr)

            if not callable(obj):
                continue

            sig = inspect.signature(obj)
            params = []
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                type_str = ''
                if param.annotation is not inspect.Parameter.empty:
                    ann = param.annotation
                    if isinstance(ann, str):
                        type_str = f": {ann}"
                    elif hasattr(ann, '__name__'):
                        type_str = f": {ann.__name__}"
                    else:
                        type_str = f": {str(ann)}"
                default_str = ''
                if param.default is not inspect.Parameter.empty:
                    default_str = f" = {repr(param.default)}"
                params.append(f"{name}{type_str}{default_str}")

            return_str = ''
            if sig.return_annotation is not inspect.Parameter.empty:
                ann = sig.return_annotation
                if isinstance(ann, str):
                    return_str = f" -> {ann}"
                elif hasattr(ann, '__name__'):
                    return_str = f" -> {ann.__name__}"
                else:
                    return_str = f" -> {str(ann)}"

            return f"{function_name}({', '.join(params)}){return_str}"

        except (ImportError, AttributeError):
            continue
        except Exception:
            continue

    return function_name
