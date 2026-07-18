from config import *
from stage_1_function import *
from generate_prompt import *
import inspect
import threading

CRASH_BUG_PATH = root_path + f'/documentation/results/{lib_name}_crash_bugs.json'

def _save_crash_bug(api_name, entry):
    """立即将库级崩溃 bug 写入 crash JSON 文件（增量合并）"""
    bug_entry = {
        "api_name": api_name,
        **entry
    }
    save_api_inputs(api_name, [bug_entry], CRASH_BUG_PATH)
'''
存储整个方法中的小步骤 

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''
def generate_api_conditions(api_names):
    # 初始化 DeepSeek 客户端
    client = make_client()

    # 读取完整定义行（含签名），与 api_names 一一对应
    with open(f"../documentation/lib_api/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    i = 0
    while(True):
        # 获取函数名
        fun_string = api_names[i]
        api_def = api_defs[i]

        # 获取函数文档字符串
        function_name = filter_samenames(i, fun_string, api_names)
        i += 1
        api_doc = get_doc(function_name)

        if api_doc == False:
            print(f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
            continue

        # 如果 APIdef.txt 行不含签名，fallback 到 inspect 反射
        if '(' not in api_def:
            sig_str = get_function_signature_str(function_name)
            if sig_str != function_name:
                print(f"[签名反射] {function_name} -> {sig_str}")
                api_def = sig_str
            else:
                print(f"[警告] 无法获取 {function_name} 的签名，使用名称")

        # 生成prompt：传入完整签名
        prompt_1 = generate_prompt_1(fun_string, api_def, api_doc)
        # print("_________________________________________________________________________________________________________")
        # print(prompt_1)
        
        outputs_text = call_llm_with_retry(
            client, MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_1},
            ]
        )

        # print("_________________________________________________________________________________________________________")
        # print(outputs_text)
        
        # handle_output 需要根据线上输出微调（API 不会带入 prompt 本身，仅输出结果）
        # 传递 model_path 可能是为了在 handle_output 中做逻辑判断，予以保留
        api_conditions = extract_clean_json(outputs_text)
        
        # print("_________________________________________________________________________________________________________")
        # print(api_conditions)
        
        # 存储至json
        path = root_path + f'/documentation/conditions/{lib_name}_conditions.json'

        # path = f'C:\\Users\\86184\\Desktop\\Papers\\documentation\\conditions\\{lib_name}_conditions.json'
        append_api_condition_to_json(path, function_name, api_conditions)
        
        print(f"进度: {i}/{len(api_names)}")
        if i >= len(api_names):
            break


def base_condition_filter(api_names):
    i = 0
    j = 0
    # 记录需要删除的 API 名称（即组合为空的 API）
    apis_to_remove = set()
    
    api_def_path = f"../documentation/lib_api/{lib_name}_APIdef.txt"

    while i < len(api_names):
        # 获取函数名
        fun_string = api_names[i]
        function_name = filter_samenames(i, fun_string, api_names)

        # 得到所有合法参数 -> 生成组合 -> 过滤组合
        args = get_all_parameters(function_name)
        all_combinations = generate_all_combinations(args)

        json_path = root_path + f'/documentation/conditions/{lib_name}_conditions.json'
        conditions = get_api_conditions(function_name, json_path)
        filtered_combinations = filter_combinations(all_combinations, conditions)

        # ==========================================
        # 核心逻辑修改：如果组合为空，记录待删除
        # ==========================================
        if not filtered_combinations:
            print(f"[-] API '{function_name}' 的参数组合过滤后为空，标记为待删除。")
            apis_to_remove.add(function_name)
        else:
            # 只有在组合不为空时，才执行存储逻辑
            path = root_path + f'/documentation/arg_combinations/{lib_name}_combinations_{j}.json'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if os.path.exists(path):
                if is_file_too_large(path, max_size_mb=10):
                    j += 1
                    path = root_path + f'/documentation/arg_combinations/{lib_name}_combinations_{j}.json'
            
            # 如果文件不存在，初始化空 JSON
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump({}, f)

            append_filtered_combinations_to_json(path, function_name, filtered_combinations)

        i += 1

    # ==========================================
    # 物理清理逻辑：覆写 APIdef.txt 文件
    # ==========================================
    if apis_to_remove:
        print(f"[*] 开始从 {api_def_path} 中物理清理无效 API...")
        if os.path.exists(api_def_path):
            with open(api_def_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            valid_lines = []
            for line in original_lines:
                original_str = line.strip()
                if not original_str: continue
                
                # 提取名称进行比对（处理可能带括号的情况）
                current_clean_name = original_str.split('(')[0]
                if current_clean_name not in apis_to_remove:
                    valid_lines.append(original_str)
            
            # 覆写文件
            with open(api_def_path, 'w', encoding='utf-8') as f:
                for v_line in valid_lines:
                    f.write(f"{v_line}\n")
            print(f"[*] 清理完成。删除了 {len(apis_to_remove)} 个 API，剩余 {len(valid_lines)} 个。")



def check_condition_filter(api_names):
    # 初始化 DeepSeek 客户端
    client = make_client()

    with open(f"../documentation/lib_api/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    i = 0   # 循环变量
    j = 0   # json文件编号

    while True:
        error_combinations = []

        # 遍历每个函数的组合，检查是否满足条件
        fun_string = api_names[i]
        api_def = ""
        for def_ in api_defs:
            if fun_string in def_:
                api_def = def_
                break
          
        function_name = fun_string
        i += 1

        result = get_all_combinations_from_json(function_name, j)
        if result == False:
            print(f"[提示] {fun_string} 未在任何组合文件中找到，跳过")
            if i >= len(api_names): break
            continue
        arg_combinations, j = result
        api_doc = get_doc(function_name)
        
        if api_doc == False:
            print(f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
            if i >= len(api_names): break
            continue

        n = 0  # 进度计数
        for arg_combination in arg_combinations:
            prompt_2 = generate_prompt_2(fun_string, arg_combination, api_def, api_doc)
            # prompt_2 = "".join(char for char in str(prompt_2) if char.isprintable() or char in "\n\t")
            # --- API 调用替代本地模型推理 ---
            outputs_text = call_llm_with_retry(
                client, MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional software testing assistant."},
                    {"role": "user", "content": prompt_2},
                ]
            )
            # print(outputs_text)

            # 处理输出并判断
            # 注意：API 返回的 outputs_text 不包含 prompt，handle_output 逻辑可能需要适配
            #error_tag = handle_output(outputs_text, model_path)
            # print(error_tag)
            if 'False' in outputs_text:
                error_combinations.append(arg_combination)
            
            n += 1
            print(f"API: {function_name} | 进度：{n}/{len(arg_combinations)}")
        # --------------------------------
        path = root_path + f'/documentation/error_combinations/error_{lib_name}_combinations.json'
        # path = f'/tmp/Momo_test/error_combinations/error_{lib_name}_combinations.json'
        append_filtered_combinations_to_json(path, function_name, error_combinations)

        if i >= len(api_names):
            break



# 剪枝后组合

def generate_api_boundary(api_names):
    # 初始化 DeepSeek 客户端
    client = make_client()

    # 移除 if lib_name == "torch" 判断，直接进入通用流程
    j = 0
    path = root_path + f'/documentation/arg_boundary/cut_{lib_name}_boundary_{j}.json'
    length_api_names = len(api_names)
    i = 0

    while i < length_api_names:
        api_inputs = []
        api_name = filter_samenames(i, api_names[i], api_names)
        print(f"API进度: {i+1}/{length_api_names} | {api_name}")

        # 读取相关的 JSON 配置
        arg_combinations = read_json_api(api_name=api_name, file_path=f"../documentation/arg_combinations/", read_mode="cut_combination")
        conditions = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
        arg_spaces = read_json_api(api_name=api_names[i], file_path=f"../documentation/arg_space/", read_mode="arg_space")

        if arg_combinations is None or arg_spaces is None:
            add_log(root_path + f"/Momo_test/{lib_name}_log.txt", api_name)
            i += 1
            continue

        length_arg_spaces = len(arg_combinations)

        for arg_combination in arg_combinations:
            combinations = arg_combination["combinations"]
            length_combinations = len(combinations)

            # 匹配参数空间 ID
            arg_space = None
            for arg_sp in arg_spaces:
                if arg_sp["id"] == arg_combination["id"]:
                    arg_space = arg_sp
                    break

            if not arg_space:
                continue

            for comb_idx, comb in enumerate(combinations):


                path_type = arg_space["path_type"]
                prompt = generate_prompt_3(api_name, comb, arg_space, conditions["Parameter type"])

                # --- 调用线上 API ---
                outputs_text = call_llm_with_retry(
                    client, MODEL,
                    messages=[
                        {"role": "system", "content": "You are a specialized AI for API boundary analysis and software testing."},
                        {"role": "user", "content": prompt},
                    ]
                )

                # 解析输出
                api_boundary = extract_clean_json(outputs_text)

                new_api_input_boundary = {"path_type": path_type, "api_input": api_boundary}
                api_inputs.append(new_api_input_boundary)

        # 存储至 JSON
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path) and is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = root_path + f'/documentation/arg_boundary/{lib_name}_boundary_{j}.json'
        
        save_api_inputs(api_name, api_inputs, path)
        
        i += 1

    return



#------------------------------------
# 生成默认输入
#------------------------------------
def generate_default_inputs(api_names):
    # 初始化 DeepSeek 客户端
    client = make_client()

    # 重新从文件读取最新的 API 列表
    api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")

    j = 0
    path = root_path + f'/documentation/api_input/{lib_name}_default_inputs_{j}.json'
    length_api_names = len(api_names)
    i = 0

    # 确保保存目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    while i < length_api_names:
        api_name = filter_samenames(i, api_names[i], api_names)
        api_doc = get_doc(api_name)
        
        # 读取该 API 的参数类型条件
        conditions = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
        
        print(f"进度: {i+1}/{length_api_names} | 正在处理 API: {api_name}")
        
        # 生成 Prompt 4
        prompt = generate_prompt_4(api_name, conditions["Parameter type"], api_doc)

        # --- 调用线上 API ---
        outputs_text = call_llm_with_retry(
            client, MODEL,
            messages=[
                {"role": "system", "content": "You are a specialized AI assistant for generating default API inputs and test cases."},
                {"role": "user", "content": prompt},
            ]
        )

        # 使用之前修改好的 extract_clean_json 抽取 JSON
        api_default_input = extract_clean_json(outputs_text)
                
        # 存储至 json
        # 检查文件大小，必要时切换文件编号
        if os.path.exists(path) and is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = root_path + f'/documentation/api_input/{lib_name}_default_inputs_{j}.json'
        
        save_api_inputs(api_name, api_default_input, path)
        
        i += 1

    return




#------------------------------------
# 生成api input
#------------------------------------

def generate_api_input_old(api_names):
    # with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
    #     api_defs = [line.strip() for line in file]
    api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")

    # 加载LLM模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )

    if lib_name == "torch":
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        j = 0
        k = 0
        path = root_path + f'/documentation/api_input/{lib_name}_inputs_{j}.json'
        length_api_names = len(api_names)
        for i in range(length_api_names):
            api_inputs = []
            api_name = api_names[i]
            api_boundarys = read_json_api(api_name=api_name, file_path=f"../documentation/arg_boundary/cut_{lib_name}_boundary_{k}.json", read_mode="boundary")
            api_default_inputs = read_json_api(api_name=api_name, file_path=f"../documentation/api_input/{lib_name}_default_inputs_{j}.json", read_mode="default_input")
            # 将api_boundary转换为字典形式
            # api_boundary = json.loads(api_boundary_str)
            n = 0
            for api_boundary in api_boundarys:
                
                api_input = generate_test_inputs_from_api_boundaries(api_name, api_boundary["api_input"], model = model, tokenizer = tokenizer, default_inputs = api_default_inputs)
                new_api_input = {"path_type": api_boundary["path_type"], "api_input": api_input}
                api_inputs.append(new_api_input)
                if n == 0:
                    break
            #存储至json
            if is_file_too_large(path, max_size_mb=1000):
                j+=1
                path = root_path + f'/documentation/api_input/{lib_name}_input_{j}.json'
                save_api_inputs(api_name, api_inputs, path)
            else:
                save_api_inputs(api_name, api_inputs, path)
            print(f"已完成{api_name}的API输入生成, 进度"+str(i)+"/"+str(len(api_names)))
            if i < 1:
                break


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


def _is_code_type(type_desc):
    """根据参数类型描述判断该参数的候选值是否需要 eval 执行"""
    code_keywords = ["Tensor", "tensor", "ndarray", "object", "callable",
                     "Callable", "module", "Optimizer", "Generator", "nn.Module"]
    return any(kw in type_desc for kw in code_keywords)


def generate_api_input(api_names):
    # 初始化 DeepSeek 客户端
    client = make_client()

    j = 0
    path = root_path + f'/documentation/api_input/{lib_name}_inputs_{j}.json'
    length_api_names = len(api_names)
    i = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    while i < length_api_names:
        api_name = api_names[i]
        api_boundarys = read_json_api(api_name=api_name, file_path=f"../documentation/arg_boundary/", read_mode="boundary")
        api_conditions = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
        api_code = read_json_api(api_name=api_name, file_path=f"../documentation/api_src_code/", read_mode="src_code")
        api_doc = get_doc(api_name)

        # 防御性检查
        if api_conditions is None or "Parameter type" not in api_conditions:
            print(f"[跳过] {api_name} 缺少 conditions，跳过")
            i += 1
            continue

        arg_dict = api_conditions["Parameter type"]
        api_inputs_candidate = {}

        for key, value in arg_dict.items():
            prompt = generate_prompt_8(api_name, key, value, api_boundarys, api_doc, api_code)
            outputs_text = call_llm_with_retry(
                client, MODEL,
                messages=[
                    {"role": "system", "content": "You are a specialized AI assistant for generating API test inputs and test cases."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                seed=42
            )

            if outputs_text is None:
                print(f"[警告] {api_name} 参数 {key} LLM 返回为空，跳过")
                continue

            arg_input = extract_clean_list(outputs_text)
            # 为每个参数打上类型标签，区分 code 字符串和 literal 字符串
            param_type = "code" if _is_code_type(value) else "literal"
            api_inputs_candidate[key] = {"type": param_type, "values": arg_input}

        # 存储至 json
        if is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = root_path + f'/documentation/api_input/{lib_name}_inputs_{j}.json'
        save_api_inputs(api_name, api_inputs_candidate, path)
        print(f"已完成{api_name}的API输入生成, 进度 {i+1}/{length_api_names}")
        i += 1
    return

#------------------------------------
# 生成测试案例model
#------------------------------------
def generate_test_cases(api_names):

    client = make_client()

    # 读取 API 定义
    with open(f"../documentation/lib_api/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")

    j = 0
    path = f"{root_path}/documentation/test_cases/{lib_name}_case_{j}.json"

    for i in range(len(api_names)):
        api_name = api_names[i]
        function_name = filter_samenames(i, api_name, api_names)
        api_def = api_defs[i]

        api_doc = get_doc(function_name)
        prompt_6 = generate_prompt_6(api_name, api_def, api_doc)

        outputs_text = call_llm_with_retry(
            client, MODEL,
            messages=[
                {"role": "system", "content": "You are a specialized AI assistant for generating API test inputs and test cases."},
                {"role": "user", "content": prompt_6},
            ],
            temperature=0.0,
            top_p=1.0,
            seed=42
        )

        case = outputs_text

        if is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = f"{root_path}/documentation/test_cases/{lib_name}_case_{j}.json"

        save_api_inputs(api_name, case, path)
        print(f"已完成 {api_name} 的API测试案例生成, 进度 {i + 1}/{len(api_names)}")

# api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")
# generate_test_cases(api_names)
#------------------------------------
# 对测试案例model注入测试输入并运行
#------------------------------------
def _extract_run_api_code(case_text):
    """从 LLM 输出中提取 run_api 的 Python 代码，兼容多种格式"""
    if case_text is None:
        return None
    # 尝试提取 markdown 代码块
    code_match = re.search(r'```python\n(.*?)\n```', case_text, re.DOTALL)
    if code_match:
        return code_match.group(1)
    # 尝试提取不带 python 标记的代码块
    code_match = re.search(r'```\n(.*?)\n```', case_text, re.DOTALL)
    if code_match:
        return code_match.group(1)
    # 如果文本以 def 开头，直接使用
    stripped = case_text.strip()
    if stripped.startswith("def "):
        return stripped
    return None


def _load_run_api(api_name):
    """从 test_cases 加载 run_api 函数，返回 callable 或 None"""
    case = read_json_api(api_name=api_name, file_path=f"../documentation/test_cases/", read_mode="case")
    if case is None:
        return None
    code_str = _extract_run_api_code(case)
    if code_str is None:
        return None
    local_namespace = {}
    try:
        exec_globals = dict(globals())
        try:
            lib_mod = importlib.import_module(lib_name)
            exec_globals[lib_name] = lib_mod
        except ImportError:
            print(f"[警告] 无法导入库 {lib_name}，run_api 可能无法正常执行")
        exec(code_str, exec_globals, local_namespace)
        return local_namespace.get("run_api")
    except Exception as e:
        print(f"解析 {api_name} 的 case 代码失败: {e}")
        return None


_valid_params_cache = {}

def _get_function_params(api_name):
    """从被测库的源码反射获取函数实际参数名列表，用于过滤 LLM 虚构的参数"""
    global _valid_params_cache
    if api_name in _valid_params_cache:
        return _valid_params_cache[api_name]
    try:
        parts = api_name.split(".")
        mod = importlib.import_module(".".join(parts[:-1]))
        func = getattr(mod, parts[-1])
        sig = inspect.signature(func)
        params = set(sig.parameters.keys())
        _valid_params_cache[api_name] = params
        return params
    except Exception:
        _valid_params_cache[api_name] = None
        return None


# 缓存 glom 命名空间，供 eval 使用
_eval_globals_cache = None


def _get_eval_globals():
    """构建包含被测库公开 API 的 eval 命名空间，避免 eval 时 NameError"""
    global _eval_globals_cache
    if _eval_globals_cache is None:
        _eval_globals_cache = {}
        try:
            lib_mod = importlib.import_module(lib_name)
            _eval_globals_cache[lib_name] = lib_mod
            for name in dir(lib_mod):
                if not name.startswith("_"):
                    try:
                        _eval_globals_cache[name] = getattr(lib_mod, name)
                    except Exception:
                        pass
        except ImportError:
            pass
        for alias in ["torch", "numpy", "np", "asyncio", "concurrent"]:
            try:
                _eval_globals_cache[alias] = importlib.import_module(alias)
            except ImportError:
                pass
    return _eval_globals_cache


def _eval_param_by_type(param_value, param_type):
    """根据类型标签决定是否 eval：code 类型 eval，literal 类型也尝试还原 repr() 序列化的非基本类型"""
    if not isinstance(param_value, str):
        return param_value

    eval_globals = _get_eval_globals()

    if param_type == "code":
        try:
            return eval(param_value, eval_globals)
        except Exception:
            return param_value

    # safe_serialize 对非基本类型调用 repr() 序列化，这里逆向还原
    stripped = param_value.strip()
    if stripped and stripped[0] in "{([":
        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            try:
                return eval(param_value, eval_globals)
            except Exception:
                return param_value

    # 部分 LLM 生成的候选值是代码表达式但被误标为 "literal"（如 "Val(0)", "Path('a','b')"）
    # 尝试 eval：只接受结果为非字符串对象，避免将普通字符串意外转换
    if stripped and "(" in stripped:
        try:
            result = eval(param_value, eval_globals)
            if not isinstance(result, str):
                return result
        except Exception:
            pass

    return param_value


class _ApiTimeoutError(Exception):
    """API 执行超时异常"""
    pass


def _run_with_timeout(run_api, timeout, **kwargs):
    """在 daemon 线程中运行 API，超时抛出 _ApiTimeoutError，不等待卡死线程"""
    result = None
    exc = None

    def target():
        nonlocal result, exc
        try:
            result = run_api(**kwargs)
        except Exception as e:
            exc = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise _ApiTimeoutError(f"API 执行超时 ({timeout}s)")
    if exc is not None:
        raise exc
    return result


def run_test_cases_v1(K=100, output_path=None):
    """
    运行测试案例框架（差分测试 V1 基准录制版）
    :param K: 每个 API 组装并执行的测试用例数量
    :param output_path: 基线输出路径，默认 results/{lib_name}_v1_baseline.json
    """
    api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")
    if output_path is None:
        output_path = root_path + f'/documentation/results/{lib_name}_v1_baseline.json'
    page_pattern = output_path.replace('.json', '_{j}.json')

    j = 0
    path = output_path
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for i, api_name in enumerate(api_names):
        function_name = filter_samenames(i, api_name, api_names)

        inputs_dict = read_json_api(api_name=api_name, file_path=f"../documentation/api_input/", read_mode="inputs")
        if inputs_dict is None:
            print(f"[跳过] {api_name} 无测试输入数据")
            continue

        run_api = _load_run_api(api_name)
        api_run_results = []
        if not run_api:
            api_run_results.append({
                "测试输入": {},
                "函数返回结果": "run_api 函数加载失败",
                "函数运行状态": "error"
            })
            j, path = save_and_paginate(api_name, api_run_results, path, root_path, lib_name, j, page_pattern)
            continue

        # --- 初始化选择频次记录器 ---
        # inputs_dict 格式: {param_name: {"type": "code"/"literal", "values": [...]}}
        selection_counts = {}
        for param_name, param_info in inputs_dict.items():
            candidate_values = param_info.get("values", []) if isinstance(param_info, dict) else param_info
            if candidate_values:
                selection_counts[param_name] = [0] * len(candidate_values)

        # 2. 组装并执行 K 次测试用例 (整块兜底: 防止任何遗漏路径的 RecursionError 导致整体崩溃)
        last_serialized_input = {}
        last_k = 0
        try:
            for k_idx in range(K):
                assembled = {}

                # --- 改进的轮盘赌算法（基于反比权重） ---
                for param_name, param_info in inputs_dict.items():
                    if isinstance(param_info, dict):
                        candidate_values = param_info.get("values", [])
                        param_type = param_info.get("type", "literal")
                    else:
                        candidate_values = param_info
                        param_type = "literal"

                    if not candidate_values:
                        continue

                    counts = selection_counts.get(param_name)
                    if counts is None:
                        continue
                    weights = [1.0 / (c + 1) for c in counts]
                    selected_idx = random.choices(range(len(candidate_values)), weights=weights, k=1)[0]
                    selection_counts[param_name][selected_idx] += 1
                    assembled[param_name] = {"value": candidate_values[selected_idx], "type": param_type}

                serialized_input = {k: safe_serialize(v["value"]) for k, v in assembled.items()}
                last_serialized_input = serialized_input
                last_k = k_idx

                result_entry = {
                    "测试输入": serialized_input,
                    "函数返回结果": None,
                    "函数运行状态": "pending"
                }

                # --- 参数解析 (从序列化后的值反序列化，与 V2 回放路径一致) ---
                evaluated_item = {}
                eval_success = True
                for param_name, entry in assembled.items():
                    try:
                        evaluated_item[param_name] = _eval_param_by_type(serialized_input[param_name], entry["type"])
                    except RecursionError:
                        eval_success = False
                        result_entry["函数运行状态"] = "recursion_bug"
                        result_entry["函数返回结果"] = f"[RECURSION_BUG] 参数 {param_name} 的 eval() 触发了递归深度超限"
                        result_entry["bug_category"] = "recursion"
                        result_entry["bug_location"] = f"eval_param:{param_name}"
                        break
                    except Exception as e:
                        eval_success = False
                        result_entry["函数运行状态"] = "error"
                        result_entry["函数返回结果"] = f"参数解析错误 ({param_name}): {str(e)}"
                        break

                if not eval_success:
                    api_run_results.append(result_entry)
                    continue

                # --- 过滤掉 LLM 虚构的参数（不在实际函数签名中）---
                valid_params = _get_function_params(api_name)
                if valid_params is not None:
                    filtered_item = {k: v for k, v in evaluated_item.items() if k in valid_params}
                else:
                    filtered_item = evaluated_item

                # --- 运行 API (5s 超时保护，超时则跳过当前 API 剩余用例) ---
                try:
                    output = _run_with_timeout(run_api, 5, **filtered_item)
                    result_entry["函数运行状态"] = "success"
                    result_entry["函数返回结果"] = safe_serialize(output)
                except _ApiTimeoutError:
                    result_entry["函数运行状态"] = "timeout"
                    result_entry["函数返回结果"] = f"[TIMEOUT] API 执行超过 5 秒，疑似死循环或死锁"
                    result_entry["bug_category"] = "timeout"
                    result_entry["bug_location"] = "api_execution"
                    api_run_results.append(result_entry)
                    print(f"[{api_name}] 第 {k_idx+1}/{K} 个用例超时，跳过剩余用例，继续下一个 API")
                    break
                except RecursionError:
                    result_entry["函数运行状态"] = "recursion_bug"
                    result_entry["函数返回结果"] = f"[RECURSION_BUG] API 执行或结果序列化时触发递归深度超限，疑似库中存在循环引用或自引用结构"
                    result_entry["bug_category"] = "recursion"
                    result_entry["bug_location"] = "api_execution"
                    api_run_results.append(result_entry)
                except Exception as e:
                    result_entry["函数运行状态"] = "error"
                    try:
                        result_entry["函数返回结果"] = f"{type(e).__name__}: {str(e)}"
                    except Exception as str_exc:
                        result_entry["函数返回结果"] = f"{type(e).__name__}: <exception str() failed>"
                        result_entry["bug_category"] = "library_bug"
                        result_entry["bug_location"] = f"{type(e).__name__}.__str__"
                        result_entry["函数运行状态"] = "library_bug"
                        result_entry["bug_detail"] = f"库异常 __str__() 崩溃: {type(str_exc).__name__}: {str_exc}"
                        _save_crash_bug(api_name, result_entry.copy())
                    api_run_results.append(result_entry)
                else:
                    api_run_results.append(result_entry)

        except RecursionError:
            api_run_results.append({
                "测试输入": last_serialized_input,
                "函数返回结果": f"[RECURSION_BUG] {api_name} 在第 {last_k + 1}/{K} 个用例时整体递归崩溃，跳过剩余用例",
                "函数运行状态": "recursion_bug",
                "bug_category": "recursion",
                "bug_location": "api_loop",
                "crashed_at_iteration": last_k
            })

        # 3. 文件切分与保存逻辑
        j, path = save_and_paginate(api_name, api_run_results, path, root_path, lib_name, j, page_pattern)
        print(f"已完成 {api_name} 的API测试 (执行 {K} 次), 进度 {i+1}/{len(api_names)}")

    # 4. V1 完成后自动提取 bug 报告
    generate_bug_report(result_path=output_path)
    generate_timeout_bug_report(result_path=output_path)



def _load_paginated_baseline(baseline_path, max_pages=20):
    """加载可能分页的基线文件，合并为单个 dict"""
    v1_baseline = {}
    # 加载主文件
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                v1_baseline.update(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"[警告] 读取基线文件失败 {baseline_path}: {e}")
    # 加载分页文件: {name}.json → {name}_1.json, {name}_2.json, ...
    base_no_ext = baseline_path.replace('.json', '')
    for j in range(1, max_pages):
        page_path = f"{base_no_ext}_{j}.json"
        if not os.path.exists(page_path):
            break
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                v1_baseline.update(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"[警告] 读取分页文件失败 {page_path}: {e}")
    return v1_baseline


def run_test_cases_v2(baseline_path=None, report_path=None):
    """
    读取 V1 基线并执行 V2 差分测试
    :param baseline_path: V1 基线 JSON 文件路径
    :param report_path: 差分报告输出路径
    """
    if baseline_path is None:
        baseline_path = root_path + f'/documentation/results/{lib_name}_v1_baseline.json'
    if report_path is None:
        report_path = root_path + f'/documentation/results/{lib_name}_diff_report.json'

    print(f"正在加载 V1 基线数据: {baseline_path} ...")
    v1_baseline = _load_paginated_baseline(baseline_path)
    if not v1_baseline:
        print(f"错误: 基线为空或文件不存在。请先运行 V1 录制。")
        return

    diff_report = []
    total_apis = len(v1_baseline)
    api_idx = 0

    for api_name, cases in v1_baseline.items():
        api_idx += 1
        print(f"正在测试 API: {api_name} ({len(cases)} 个用例) [{api_idx}/{total_apis}]")

        # 1. 加载 run_api
        run_api = _load_run_api(api_name)
        if not run_api:
            print(f"[{api_name}] 未找到 run_api，跳过测试。")
            continue

        # 2. 读取 inputs_dict 获取参数类型标签 (code/literal)
        inputs_dict = read_json_api(api_name=api_name, file_path=f"../documentation/api_input/", read_mode="inputs")

        # 3. 遍历 V1 记录的每个用例 (外层兜底: 防止遗漏的 RecursionError 导致整个 API 崩溃)
        try:
            for idx, case in enumerate(cases):
                inputs_str_dict = case.get("测试输入", {})
                v1_result = case.get("函数返回结果")
                v1_status = case.get("函数运行状态")

                # --- 反序列化输入参数：根据类型标签决定是否 eval ---
                evaluated_item = {}
                eval_success = True
                for k, v in inputs_str_dict.items():
                    # 从 inputs_dict 获取参数类型
                    param_type = "literal"
                    if inputs_dict and k in inputs_dict:
                        param_info = inputs_dict[k]
                        if isinstance(param_info, dict):
                            param_type = param_info.get("type", "literal")

                    try:
                        evaluated_item[k] = _eval_param_by_type(v, param_type)
                    except RecursionError:
                        eval_success = False
                        break
                    except Exception:
                        eval_success = False
                        break

                if not eval_success:
                    continue

                # --- 过滤掉 LLM 虚构的参数（不在实际函数签名中）---
                valid_params = _get_function_params(api_name)
                if valid_params is not None:
                    filtered_item = {k: v for k, v in evaluated_item.items() if k in valid_params}
                else:
                    filtered_item = evaluated_item

                # --- 运行 V2 API (5s 超时保护) ---
                v2_result = None
                v2_status = "pending"
                try:
                    output = _run_with_timeout(run_api, 5, **filtered_item)
                    v2_result = safe_serialize(output)
                    v2_status = "success"
                except _ApiTimeoutError:
                    v2_result = f"[TIMEOUT] API 执行超过 5 秒，疑似死循环或死锁"
                    v2_status = "timeout"
                except RecursionError:
                    v2_result = f"[RECURSION_BUG] API 执行或结果序列化时触发递归深度超限，疑似库中存在循环引用或自引用结构"
                    v2_status = "recursion_bug"
                except Exception as e:
                    try:
                        v2_result = f"{type(e).__name__}: {str(e)}"
                    except Exception as str_exc:
                        v2_result = f"{type(e).__name__}: <exception str() failed> [library_bug: {type(str_exc).__name__}]"
                        _save_crash_bug(api_name, {
                            "函数返回结果": v2_result,
                            "函数运行状态": "library_bug",
                            "bug_category": "library_bug",
                            "bug_location": f"{type(e).__name__}.__str__",
                            "bug_detail": f"库异常 __str__() 崩溃: {type(str_exc).__name__}: {str_exc}",
                            "version": "v2"
                        })
                    v2_status = "error"

                # --- 差分断言 ---
                is_identical, reason = compare_results(v1_result, v2_result, v1_status, v2_status)

                if not is_identical:
                    diff_report.append({
                        "api_name": api_name,
                        "case_index": idx,
                        "inputs": inputs_str_dict,
                        "v1_status": v1_status,
                        "v2_status": v2_status,
                        "v1_output": v1_result,
                        "v2_output": v2_result,
                        "diff_reason": reason
                    })

                if v2_status == "timeout":
                    print(f"[{api_name}] 第 {idx+1} 个用例超时，跳过剩余用例，继续下一个 API")
                    break

        except RecursionError:
            print(f"[{api_name}] 用例循环整体递归崩溃，跳过剩余用例，继续下一个 API")

    # 4. 输出差分报告
    print("\n" + "=" * 50)
    total_cases = sum(len(cases) for cases in v1_baseline.values())
    if diff_report:
        print(f"测试完毕。{len(diff_report)}/{total_cases} 个用例存在版本行为差异！")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(diff_report, f, ensure_ascii=False, indent=4)
        print(f"差分报告已保存至: {report_path}")
    else:
        print(f"测试通过：{total_cases} 个用例均未发现版本行为差异，向下兼容。")


# ------------------------------------
# 统一测试入口
# ------------------------------------
def run_test_cases(K=100):
    """
    差分测试统一入口。
    自动检测 V1 基线是否存在：
    - 无基线 → V1 模式：录制基准数据
    - 有基线 → V2 模式：执行差分测试并生成报告
    """
    baseline_path = root_path + f'/documentation/results/{lib_name}_v1_baseline.json'
    report_path = root_path + f'/documentation/results/{lib_name}_diff_report.json'

    if os.path.exists(baseline_path):
        print("=" * 50)
        print("[V2 模式] 检测到 V1 基线，运行差分测试...")
        print("=" * 50)
        run_test_cases_v2(baseline_path=baseline_path, report_path=report_path)
    else:
        print("=" * 50)
        print("[V1 模式] 未检测到基线，运行基准录制...")
        print("=" * 50)
        run_test_cases_v1(K=K, output_path=baseline_path)
        # 自动生成递归 bug 汇总报告
        generate_bug_report(result_path=baseline_path)
