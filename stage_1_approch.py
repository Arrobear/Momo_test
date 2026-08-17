from config import *
from stage_1_function import *
from generate_prompt import *
import inspect
import sys
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
        parameter_names, required_parameters = extract_signature_parameters(api_def)
        if not isinstance(api_conditions, dict):
            api_conditions = {}
        parameter_types = api_conditions.get("Parameter type")
        used_signature_fallback = not isinstance(parameter_types, dict) or not parameter_types
        if used_signature_fallback:
            print(f"[条件回退] {function_name} 使用函数签名恢复参数列表")
            api_conditions["Parameter type"] = {
                parameter_name: "unknown"
                for parameter_name in parameter_names
            }
        mandatory_parameters = api_conditions.get("Mandatory Parameters")
        if (
            not isinstance(mandatory_parameters, list)
            or used_signature_fallback and not mandatory_parameters
        ):
            api_conditions["Mandatory Parameters"] = required_parameters
        for condition_key in (
            "Mutually Exclusive Parameter Pairs",
            "Mandatory Coexistence Parameters",
            "Conditional Mutual Exclusion Parameters",
        ):
            if not isinstance(api_conditions.get(condition_key), list):
                api_conditions[condition_key] = []
        
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
            mandatory = [
                parameter for parameter in conditions.get("Mandatory Parameters", [])
                if parameter in args
            ]
            fallback = mandatory or list(args)
            filtered_combinations = [fallback]
            print(f"[组合回退] {function_name} 保留签名组合: {fallback}")

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

        if (
            arg_combinations is None
            or arg_spaces is None
            or not isinstance(conditions, dict)
            or "Parameter type" not in conditions
        ):
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
        if not isinstance(conditions, dict) or "Parameter type" not in conditions:
            print(f"[跳过] {api_name} 缺少参数条件，无法生成默认输入")
            i += 1
            continue
        
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


def _is_code_value(value):
    """判断单个候选值字符串是否看起来像需要 eval 的代码表达式。

    匹配模式：ClassName(...) 构造器调用、module.function(...) 调用等。
    用于兜底 _is_code_type 漏掉的类型（如 LLM 为参数类型描述不包含 code 关键字的参数
    生成了构造器表达式）。
    """
    if not isinstance(value, str):
        return False
    # 匹配 ClassName(...) 或 module.ClassName(...) 或 module.func(...) 调用模式
    import re
    return bool(re.match(
        r'^[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)*\(.*\)$',
        value.strip()
    )) and not _looks_like_literal(value)


def _looks_like_literal(value):
    """排除看起来像 Python 字面量而非构造器调用的表达式。
    例如：dict(...), list(...), tuple(...), set(...), int(...), str(...), float(...), bool(...)
    """
    builtin_calls = {'dict', 'list', 'tuple', 'set', 'frozenset',
                     'int', 'str', 'float', 'bool', 'bytes', 'bytearray',
                     'complex', 'chr', 'ord', 'hex', 'oct', 'bin', 'repr',
                     'len', 'abs', 'min', 'max', 'sum', 'sorted', 'reversed',
                     'enumerate', 'zip', 'map', 'filter', 'iter', 'next',
                     'object', 'super', 'slice', 'range', 'memoryview'}
    first_paren = value.find('(')
    if first_paren == -1:
        return False
    name = value[:first_paren].strip()
    # 去掉模块前缀
    base = name.split('.')[-1] if '.' in name else name
    return base in builtin_calls


def _any_value_is_code(values):
    """检查候选值列表中是否有任何一个看起来像代码表达式"""
    if not values:
        return False
    return any(_is_code_value(v) for v in values)


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
        if not isinstance(api_boundarys, list):
            api_boundarys = []
        if api_code is None:
            api_code = {}

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
            # 先按参数类型描述判断，再检查候选值本身是否像构造器调用（兜底）
            param_type = "code" if (_is_code_type(value) or _any_value_is_code(arg_input)) else "literal"
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
def _extract_path_test_case_payload(outputs_text):
    payload = extract_clean_json(outputs_text) if outputs_text else None
    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        return {
            "code": payload["code"].strip(),
            "summary": str(payload.get("summary", "")).strip(),
        }

    if isinstance(outputs_text, str):
        code_match = re.search(
            r"```(?:python)?\s*(.*?)```",
            outputs_text,
            re.DOTALL | re.IGNORECASE,
        )
        candidate = (
            code_match.group(1).strip()
            if code_match
            else outputs_text.strip()
        )
        if re.search(r"^\s*def\s+run_test_case\s*\(", candidate, re.MULTILINE):
            return {"code": candidate, "summary": ""}
    return None


def _read_text_if_exists(path, max_chars=12000):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def _load_bugsinpy_context(documentation_dir):
    bug_id = os.environ.get("MOMO_BUG_ID", "").strip()
    bug_dir = os.environ.get("MOMO_BUG_DIR", "").strip()
    if bug_dir:
        bug_path = Path(bug_dir)
    elif bug_id:
        bug_path = (
            Path(documentation_dir)
            / "database"
            / "BugsInPy"
            / "projects"
            / lib_gitname
            / "bugs"
            / bug_id
        )
    else:
        return {}

    return {
        "bug_id": bug_id,
        "bug_dir": str(bug_path),
        "bug_patch": _read_text_if_exists(bug_path / "bug_patch.txt"),
        "run_test": _read_text_if_exists(bug_path / "run_test.sh"),
        "bug_info": _read_text_if_exists(bug_path / "bug.info"),
    }


def generate_test_cases(api_names, k=1):

    client = make_client()
    if k <= 0:
        raise ValueError("k must be greater than zero")

    documentation_dir = Path(root_path) / "documentation"
    api_def_path = documentation_dir / "lib_api" / f"{lib_name}_APIdef.txt"

    # 读取 API 定义
    with open(api_def_path, 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    api_names = read_file(api_def_path)

    j = 0
    path = str(documentation_dir / "test_cases" / f"{lib_name}_case_{j}.json")
    required_python = os.environ.get("MOMO_REQUIRED_PYTHON", "unknown")
    bug_context = _load_bugsinpy_context(documentation_dir)

    for i in range(len(api_names)):
        api_name = api_names[i]
        function_name = filter_samenames(i, api_name, api_names)
        api_def = api_defs[i]

        api_doc = get_doc(function_name)
        api_code = read_json_api(
            api_name=api_name,
            file_path=documentation_dir / "api_src_code",
            read_mode="src_code",
        ) or {}
        conditions = read_json_api(
            api_name=api_name,
            file_path=documentation_dir / "conditions",
            read_mode="conditions",
        ) or {}
        api_boundaries = read_json_api(
            api_name=api_name,
            file_path=documentation_dir / "arg_boundary",
            read_mode="boundary",
        ) or []
        arg_spaces = read_json_api(
            api_name=api_name,
            file_path=documentation_dir / "arg_space",
            read_mode="arg_space",
        )
        if not isinstance(arg_spaces, list) or not arg_spaces:
            arg_spaces = [{
                "id": f"{api_name}_P1",
                "conjuncts": [],
                "path_type": "return",
                "src": [],
            }]

        path_cases = []
        for path_index, path_data in enumerate(arg_spaces):
            if not isinstance(path_data, dict):
                continue
            path_id = str(path_data.get("id") or f"{api_name}_P{path_index + 1}")
            path_type = str(path_data.get("path_type", "return"))
            expected_status = "error" if path_type == "raise" else "success"

            for sample_index in range(k):
                prompt = generate_prompt_9(
                    api_name=api_name,
                    api_signature=api_def,
                    api_doc=api_doc,
                    api_code=api_code,
                    conditions=conditions,
                    api_boundaries=api_boundaries,
                    bug_context=bug_context,
                    path_data=path_data,
                    sample_index=sample_index,
                    required_python=required_python,
                )
                outputs_text = call_llm_with_retry(
                    client,
                    MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Generate executable path-specific Python tests. "
                                "Output JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    seed=42 + sample_index,
                )
                payload = _extract_path_test_case_payload(outputs_text)
                if payload is None:
                    payload = {
                        "code": (
                            "def run_test_case():\n"
                            "    raise RuntimeError("
                            "'initial LLM output did not contain a runnable test')"
                        ),
                        "summary": "Initial generation could not be parsed.",
                    }

                path_cases.append({
                    "schema_version": 2,
                    "case_id": f"{path_id}::case_{sample_index + 1}",
                    "api_name": api_name,
                    "api_signature": api_def,
                    "required_python": required_python,
                    "api_documentation": api_doc,
                    "api_source": api_code,
                    "parameter_conditions": conditions,
                    "boundary_context": api_boundaries,
                    "bug_context": bug_context,
                    "path_id": path_id,
                    "path_type": path_type,
                    "path_constraints": path_data.get("conjuncts", []),
                    "path_source": path_data.get("src", []),
                    "expected_status": expected_status,
                    "code": payload["code"],
                    "summary": payload["summary"],
                    "revision": 0,
                    "validated": False,
                    "validation_history": [],
                })

        if is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = str(
                documentation_dir
                / "test_cases"
                / f"{lib_name}_case_{j}.json"
            )

        save_api_inputs(api_name, path_cases, path)
        print(
            f"已完成 {api_name} 的路径测试案例生成: "
            f"{len(arg_spaces)} 条路径 x {k}, "
            f"进度 {i + 1}/{len(api_names)}"
        )

# api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")
# generate_test_cases(api_names)
#------------------------------------
# 对测试案例model注入测试输入并运行
#------------------------------------
def _extract_run_api_code(case_text):
    """从 LLM 输出中提取 run_api 的 Python 代码，兼容多种格式"""
    if case_text is None:
        return None
    if isinstance(case_text, dict):
        for value in case_text.values():
            code = _extract_run_api_code(value)
            if code and re.search(r"^\s*def\s+run_api\s*\(", code, re.MULTILINE):
                return code
        return None
    if isinstance(case_text, list):
        for value in case_text:
            code = _extract_run_api_code(value)
            if code:
                return code
        return None
    if not isinstance(case_text, str):
        return None
    # 尝试提取 markdown 代码块
    code_match = re.search(r'```(?:python)?\s*(.*?)```', case_text, re.DOTALL | re.IGNORECASE)
    if code_match:
        candidate = code_match.group(1).strip()
        if re.search(r"^\s*def\s+run_api\s*\(", candidate, re.MULTILINE):
            return candidate
    # 如果文本以 def 开头，直接使用
    stripped = case_text.strip()
    if re.search(r"^def\s+run_api\s*\(", stripped):
        return stripped
    return None


def _resolve_api_callable(api_name):
    """Resolve an API by importing the longest module prefix."""
    parts = api_name.split(".")
    for module_end in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:module_end])
        try:
            target = importlib.import_module(module_name)
        except ImportError:
            continue
        try:
            for attribute in parts[module_end:]:
                target = getattr(target, attribute)
        except AttributeError:
            continue
        if callable(target):
            return target
    return None


def _load_run_api(api_name):
    """从 test_cases 加载 run_api 函数，返回 callable 或 None"""
    case = read_json_api(api_name=api_name, file_path=f"../documentation/test_cases/", read_mode="case")
    code_str = _extract_run_api_code(case)
    if code_str is not None:
        try:
            exec_globals = dict(globals())
            # Dataset API names may include repository-layout prefixes such as
            # "lib.ansible...". Import the longest valid module path and expose
            # its root package so generated code can resolve the exact API name.
            api_parts = api_name.split(".")
            for end in range(len(api_parts) - 1, 0, -1):
                try:
                    importlib.import_module(".".join(api_parts[:end]))
                    exec_globals[api_parts[0]] = importlib.import_module(api_parts[0])
                    break
                except ImportError:
                    continue
            try:
                lib_mod = importlib.import_module(lib_name)
                exec_globals[lib_name] = lib_mod
            except ImportError:
                pass
            exec(code_str, exec_globals)
            generated = exec_globals.get("run_api")
            if callable(generated):
                return generated
        except Exception as e:
            print(f"解析 {api_name} 的 case 代码失败: {e}")

    target = _resolve_api_callable(api_name)
    if target is None:
        print(f"[错误] 无法解析 {api_name} 的生成模板或真实 callable")
        return None

    print(f"[模板回退] {api_name} 使用真实 callable 直接执行")

    def run_api(*args, **kwargs):
        return target(*args, **kwargs)

    return run_api


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
        _eval_globals_cache = {"__builtins__": __builtins__}
        try:
            lib_mod = importlib.import_module(lib_name)
            _eval_globals_cache[lib_name] = lib_mod
            _eval_globals_cache["_momo_lib_mod"] = lib_mod
            _eval_globals_cache["_momo_lib_name"] = lib_name
            # 遍历库的所有子模块和嵌套类，注册到命名空间
            _collect_lib_symbols(lib_mod, lib_name)
            # 为常见命名差异添加别名（如文档中的 Mode → 实际类 FileMode）
            _add_common_aliases()
        except ImportError:
            pass
        for alias in ["torch", "numpy", "np", "asyncio", "concurrent"]:
            try:
                _eval_globals_cache[alias] = importlib.import_module(alias)
            except ImportError:
                pass
    return _eval_globals_cache


def _resolve_name(name, eval_globals):
    """懒解析未注册的符号名：先尝试被测库模块，再尝试作为 Python 标准库/第三方模块 import。

    处理 LLM 生成的代码中引用了库的类/函数但没有被 _collect_lib_symbols 注册的情况
    （例如库的某个内部类没有通过 dir() 暴露，或者 LLM 用了别名如 token.NAME）。
    返回解析到的对象，失败返回 None。
    """
    if name in eval_globals:
        return eval_globals[name]

    lib_mod = eval_globals.get("_momo_lib_mod")
    lib_name_val = eval_globals.get("_momo_lib_name", "")

    # 1. 直接尝试从库模块获取
    if lib_mod is not None and hasattr(lib_mod, name):
        obj = getattr(lib_mod, name)
        eval_globals[name] = obj
        return obj

    # 2. 遍历已注册的子模块，查找 name
    if lib_mod is not None:
        for key, val in eval_globals.items():
            if key.startswith(f"{lib_name_val}.") and isinstance(val, type(sys)):
                if hasattr(val, name):
                    obj = getattr(val, name)
                    eval_globals[name] = obj
                    return obj

    # 3. 重新 import 被测库并全量遍历 dir 查找
    if lib_name_val:
        try:
            lib_mod_fresh = importlib.import_module(lib_name_val)
            if hasattr(lib_mod_fresh, name):
                obj = getattr(lib_mod_fresh, name)
                eval_globals[name] = obj
                return obj
        except ImportError:
            pass

    # 4. 作为 Python 模块 import（处理 token, enum, collections, typing 等标准库/第三方库）
    try:
        mod = importlib.import_module(name)
        eval_globals[name] = mod
        return mod
    except ImportError:
        pass

    return None


class _LazyResolveDict(dict):
    """一个 dict 子类，在 __getitem__ 找不到 key 时自动尝试从被测库模块解析符号名。

    用于 eval() 的 globals — 当 eval 遇到未知变量时触发 NameError，
    在 try/except 中捕获后再用 _resolve_name 解析并重试。
    """
    pass


def _eval_with_lazy_resolve(expr, eval_globals):
    """带懒解析的 eval：NameError 时尝试从被测库动态查找缺失的符号并重试。

    最多重试 5 次（每次解析一个缺失符号），避免无限循环。
    返回值: (result, success, error_msg)
      - success=True: result 是 eval 结果
      - success=False: error_msg 描述失败原因
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = eval(expr, eval_globals)
            return result, True, None
        except NameError as e:
            import re
            m = re.search(r"name '(\w+)' is not defined", str(e))
            if not m:
                return None, False, f"NameError 但无法提取符号名: {e}"
            missing = m.group(1)
            resolved = _resolve_name(missing, eval_globals)
            if resolved is None:
                return None, False, f"无法解析符号 '{missing}'（非被测库符号，也非可导入模块）"
            # 解析成功，继续重试
        except Exception as e:
            return None, False, f"{type(e).__name__}: {str(e)[:300]}"


def _add_common_aliases():
    """为常见的命名差异添加别名映射，如 FileMode → Mode"""
    lib_mod = _eval_globals_cache.get(lib_name)
    alias_map = {}
    for name, obj in list(_eval_globals_cache.items()):
        if not isinstance(obj, type):
            continue
        # 处理带前缀的类名：如 FileMode → Mode, WriteBack → Back
        for prefix in ["File", "Write", "Read", "Parse", "Input", "Output"]:
            if name.startswith(prefix) and len(name) > len(prefix):
                short = name[len(prefix):]
                # 只在 short 不为空且首字母仍大写的纯字母名上创建别名
                if short and short[0].isupper() and short.isidentifier():
                    alias_map.setdefault(short, obj)
    for alias, obj in alias_map.items():
        if alias not in _eval_globals_cache:
            _eval_globals_cache[alias] = obj
        # 同时在库模块上设置该别名，使 black.Mode 也能工作
        if lib_mod is not None and not hasattr(lib_mod, alias):
            try:
                setattr(lib_mod, alias, obj)
            except Exception:
                pass


def _collect_lib_symbols(root_mod, root_name):
    """递归收集库的子模块和公开符号到 eval 命名空间"""
    seen = set()
    _collect_recursive(root_mod, root_name, seen)


def _collect_recursive(mod, prefix, seen):
    """递归遍历模块，注册子模块和公开类/函数"""
    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        key = f"{prefix}.{name}"
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        _eval_globals_cache[name] = obj
        _eval_globals_cache[key] = obj

        if isinstance(obj, type):
            # 注册类的嵌套类
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr_obj = getattr(obj, attr_name)
                    if id(attr_obj) not in seen:
                        seen.add(id(attr_obj))
                        _eval_globals_cache[attr_name] = attr_obj
                        _eval_globals_cache[f"{key}.{attr_name}"] = attr_obj
                except Exception:
                    pass
        elif _is_package_or_module(obj, name):
            try:
                _collect_recursive(obj, key, seen)
            except Exception:
                pass


def _is_package_or_module(obj, name):
    """判断对象是否为子模块/子包"""
    if isinstance(obj, type(sys)):
        return True
    if hasattr(obj, "__path__"):
        return True
    if hasattr(obj, "__file__") and hasattr(obj, "__name__"):
        return True
    return False


def _eval_param_by_type(param_value, param_type):
    """根据类型标签决定是否 eval：code 类型 eval，literal 类型也尝试还原 repr() 序列化的非基本类型

    返回值: (result, success) — success=False 表示 eval 失败，调用方应跳过该用例而非将字符串传入 run_api
    """
    if not isinstance(param_value, str):
        return param_value, True

    eval_globals = _get_eval_globals()
    _stripped = param_value.strip()

    if param_type == "code":
        return _eval_code_param(param_value, _stripped, eval_globals)

    # 否则按 literal 处理
    return _eval_literal_param(_stripped, eval_globals), True


def _eval_code_param(original, stripped, eval_globals):
    """eval code 类型参数，失败时尝试 AST 修复（移除无效 kwarg / 注入缺失符号）"""
    # 1. 直接 eval
    result, ok = _try_eval_code(stripped, eval_globals)
    if ok:
        return result, True

    # 2. 尝试 AST 修复
    fixed_expr = _ast_fix_call(stripped, eval_globals)
    if fixed_expr is not None and fixed_expr != stripped:
        result, ok = _try_eval_code(fixed_expr, eval_globals)
        if ok:
            import warnings
            warnings.warn(f"[eval] AST 修复后成功: {stripped[:80]}... -> {fixed_expr[:80]}...")
            return result, True

    import warnings
    warnings.warn(f"[eval] code 类型参数 eval 失败且无法修复: {original[:120]}...")
    return original, False


def _try_eval_code(expr, eval_globals):
    """尝试 eval，带 NameError 懒解析。成功且结果为非字符串→(result, True)，否则→(result, False)"""
    result, ok, _err = _eval_with_lazy_resolve(expr, eval_globals)
    if ok:
        if isinstance(result, str) and _looks_like_code_expr(expr.strip()):
            return result, False
        return result, True
    return None, False


def _ast_fix_call(expr, eval_globals):
    """AST 分析并修复常见的 LLM 表达式错误：
    1. 移除目标类/函数签名中不存在的关键字参数
    2. 将目标类型中不存在的属性访问替换为已知别名

    返回修复后的表达式字符串，无需修复则返回 None
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None

    # 收集所有需要修复的 Call 节点
    calls_to_fix = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _resolve_callable(node.func, eval_globals)
            if target is not None:
                calls_to_fix.append((node, target))

    if not calls_to_fix:
        return None

    changed = False
    for node, target in calls_to_fix:
        changed |= _drop_invalid_kwargs(node, target)

    if not changed:
        return None

    # 反序列化回源代码
    try:
        fixed = ast.unparse(tree)
    except AttributeError:
        fixed = _unparse_expr(tree.body)
    return fixed


def _resolve_callable(func_node, eval_globals):
    """尝试解析 AST 调用目标为实际可调用对象，返回 None 表示无法解析"""
    try:
        if isinstance(func_node, ast.Name):
            obj = eval_globals.get(func_node.id)
            return obj if callable(obj) else None
        elif isinstance(func_node, ast.Attribute):
            # 递归解析 a.b.c
            inner = _resolve_callable(func_node.value, eval_globals)
            if inner is not None and hasattr(inner, func_node.attr):
                return getattr(inner, func_node.attr)
            # 也尝试直接通过 eval_globals 中的全限定名查找
            code = ast.unparse(func_node) if hasattr(ast, 'unparse') else _unparse_expr(func_node)
            return eval_globals.get(code)
    except Exception:
        pass
    return None


def _drop_invalid_kwargs(call_node, target):
    """移除调用中目标签名不接受的 keyword 参数，返回是否修改。

    如果目标接受 **kwargs（变长关键字参数），则不删除任何参数。
    """
    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return False

    # 如果签名中有 **kwargs 参数，所有 keyword 参数都可能合法，不做删除
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_varkw:
        return False

    valid_params = set(sig.parameters.keys())
    # 移除 self/cls (bound method 不需要)
    valid_params.discard('self')
    valid_params.discard('cls')

    new_keywords = []
    changed = False
    for kw in call_node.keywords:
        if kw.arg is None:
            # **kwargs 展开，保留
            new_keywords.append(kw)
        elif kw.arg in valid_params:
            new_keywords.append(kw)
        else:
            changed = True

    if changed:
        call_node.keywords = new_keywords
    return changed


def _unparse_expr(node):
    """AST 表达式反序列化为源代码 (兼容 Python 3.8-)"""
    # 使用内置 compile 的方式来获取源码... 不，直接用递归还原
    return _Unparser().visit(node)


class _Unparser:
    """极简 AST → 源码转换器，覆盖常见表达式节点"""

    def visit(self, node):
        m = getattr(self, f'_un_{type(node).__name__}', None)
        if m is None:
            # fallback: 尝试用 compile 还原（仅 Python 3.9+）
            try:
                return ast.unparse(node)
            except AttributeError:
                return str(node)
        return m(node)

    def _un_Name(self, n): return n.id
    def _un_Constant(self, n):
        if isinstance(n.value, str):
            return repr(n.value)
        return str(n.value) if n.value is not ... else '...'
    def _un_Num(self, n): return str(n.n)
    def _un_Str(self, n): return repr(n.s)
    def _un_Bytes(self, n): return repr(n.s)
    def _un_Attribute(self, n): return f'{self.visit(n.value)}.{n.attr}'
    def _un_Subscript(self, n): return f'{self.visit(n.value)}[{self.visit(n.slice)}]'
    def _un_Index(self, n): return self.visit(n.value)
    def _un_Slice(self, n):
        parts = []
        if n.lower: parts.append(self.visit(n.lower))
        parts.append(':')
        if n.upper: parts.append(self.visit(n.upper))
        if n.step: parts.append(f':{self.visit(n.step)}')
        return ''.join(parts)
    def _un_List(self, n): return f'[{", ".join(self.visit(e) for e in n.elts)}]'
    def _un_Tuple(self, n):
        elts = ', '.join(self.visit(e) for e in n.elts)
        if len(n.elts) == 1: elts += ','
        return f'({elts})'
    def _un_Dict(self, n):
        return '{' + ', '.join(f'{self.visit(k)}: {self.visit(v)}' for k, v in zip(n.keys, n.values)) + '}'
    def _un_Set(self, n): return '{' + ', '.join(self.visit(e) for e in n.elts) + '}'
    def _un_Call(self, n):
        args = [self.visit(a) for a in n.args]
        args += [f'{k.arg}={self.visit(k.value)}' if k.arg else f'**{self.visit(k.value)}' for k in n.keywords]
        return f'{self.visit(n.func)}({", ".join(args)})'
    def _un_Lambda(self, n):
        args = self.visit(n.args)
        return f'lambda {args}: {self.visit(n.body)}'
    def _un_arguments(self, n):
        return ', '.join(a.arg for a in n.args)
    def _un_UnaryOp(self, n):
        ops = {ast.USub: '-', ast.UAdd: '+', ast.Not: 'not ', ast.Invert: '~'}
        return f'{ops.get(type(n.op), "?")}{self.visit(n.operand)}'
    def _un_BinOp(self, n):
        ops = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.FloorDiv: '//',
               ast.Mod: '%', ast.Pow: '**', ast.LShift: '<<', ast.RShift: '>>',
               ast.BitOr: '|', ast.BitAnd: '&', ast.BitXor: '^'}
        op = ops.get(type(n.op), '?')
        return f'({self.visit(n.left)} {op} {self.visit(n.right)})'
    def _un_Compare(self, n):
        ops = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
               ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in'}
        left = self.visit(n.left)
        parts = []
        for op, comp in zip(n.ops, n.comparators):
            parts.append(f'{ops.get(type(op), "?")} {self.visit(comp)}')
        return f'({left} {" ".join(parts)})'
    def _un_BoolOp(self, n):
        op = ' and ' if isinstance(n.op, ast.And) else ' or '
        return f'({op.join(self.visit(v) for v in n.values)})'
    def _un_IfExp(self, n):
        return f'{self.visit(n.body)} if {self.visit(n.test)} else {self.visit(n.orelse)}'
    def _un_JoinedStr(self, n):
        parts = []
        for v in n.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append('{' + self.visit(v.value) + '}')
        return "f'" + ''.join(parts) + "'"
    def _un_FormattedValue(self, n): return self.visit(n.value)
    def _un_Starred(self, n): return f'*{self.visit(n.value)}'
    def _un_ListComp(self, n):
        gens = ' '.join(self._un_comprehension(g) for g in n.generators)
        return f'[{self.visit(n.elt)} {gens}]'
    def _un_GeneratorExp(self, n):
        gens = ' '.join(self._un_comprehension(g) for g in n.generators)
        return f'({self.visit(n.elt)} {gens})'
    def _un_comprehension(self, n):
        ifs = ''.join(f' if {self.visit(f)}' for f in n.ifs)
        return f'for {self.visit(n.target)} in {self.visit(n.iter)}{ifs}'


def _eval_literal_param(stripped, eval_globals):
    """处理 literal 类型参数的反序列化"""
    # safe_serialize 对非基本类型调用 repr() 序列化，这里逆向还原
    if stripped and stripped[0] in "{([":
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            try:
                return eval(stripped, eval_globals)
            except Exception:
                return stripped

    # 部分 LLM 生成的候选值是代码表达式但被误标为 "literal"
    # 匹配 ClassName(...) / module.ClassName(...) 构造器调用模式
    if stripped and _is_code_value(stripped):
        result, ok, _err = _eval_with_lazy_resolve(stripped, eval_globals)
        if ok and not isinstance(result, str):
            return result

    # 旧版兜底：包含 () 的通用表达式
    if stripped and "(" in stripped:
        result, ok, _err = _eval_with_lazy_resolve(stripped, eval_globals)
        if ok and not isinstance(result, str):
            return result

    return stripped


def _looks_like_code_expr(s):
    """判断字符串是否看起来像代码表达式（包含函数调用或构造器）"""
    return "(" in s and ")" in s


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
                        result, ok = _eval_param_by_type(serialized_input[param_name], entry["type"])
                        if not ok:
                            eval_success = False
                            result_entry["函数运行状态"] = "eval_failed"
                            result_entry["函数返回结果"] = f"[EVAL_FAILED] 参数 {param_name} 无法反序列化: {str(serialized_input[param_name])[:200]}"
                            break
                        evaluated_item[param_name] = result
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
        raise FileNotFoundError(f"基线为空或文件不存在，请先运行 V1 录制: {baseline_path}")

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
                        result, ok = _eval_param_by_type(v, param_type)
                        if not ok:
                            eval_success = False
                            v2_eval_error = f"[EVAL_FAILED] 参数 {k} 无法反序列化: {str(v)[:200]}"
                            break
                        evaluated_item[k] = result
                    except RecursionError:
                        eval_success = False
                        v2_eval_error = f"[RECURSION_BUG] 参数 {k} 的 eval() 触发了递归深度超限"
                        break
                    except Exception as e:
                        eval_success = False
                        v2_eval_error = f"[EVAL_FAILED] 参数 {k} 解析异常: {str(e)[:200]}"
                        break

                if not eval_success:
                    # V1 的这条用例 eval 就失败了，V2 也无法 eval，但需要报告差异
                    # 如果 V1 状态也是 eval_failed/recursion_bug，说明是数据源问题而非版本差异
                    if v1_status in ("eval_failed", "recursion_bug"):
                        # V1 基线中的无效用例，记录为 SKIPPED（无法对比）而非 diff
                        pass
                    else:
                        # V1 能执行成功但 V2 eval 失败 → 可能是基线数据版本问题
                        diff_report.append({
                            "api_name": api_name,
                            "case_index": idx,
                            "inputs": inputs_str_dict,
                            "v1_status": v1_status,
                            "v2_status": "eval_failed",
                            "v1_output": v1_result,
                            "v2_output": v2_eval_error,
                            "diff_reason": f"V2 无法解析输入，V1 状态={v1_status}"
                        })
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
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(diff_report, f, ensure_ascii=False, indent=4)

    if diff_report:
        print(f"测试完毕。{len(diff_report)}/{total_cases} 个用例存在版本行为差异！")
        print(f"差分报告已保存至: {report_path}")
    else:
        print(f"测试通过：{total_cases} 个用例均未发现版本行为差异，向下兼容。")
        print(f"空差分报告已保存至: {report_path}")


# ------------------------------------
# 统一测试入口
# ------------------------------------
def run_test_cases(K=100, mode="auto"):
    """
    差分测试统一入口。
    mode="auto" 时按基线是否存在自动切换；批处理应显式传入 v1/v2，
    避免中断后残留文件改变执行语义。
    """
    baseline_path = root_path + f'/documentation/results/{lib_name}_v1_baseline.json'
    report_path = root_path + f'/documentation/results/{lib_name}_diff_report.json'

    if mode not in ("auto", "v1", "v2"):
        raise ValueError(f"未知测试模式: {mode}")

    selected_mode = mode
    if selected_mode == "auto":
        selected_mode = "v2" if os.path.exists(baseline_path) else "v1"

    if selected_mode == "v2":
        print("=" * 50)
        print("[V2 模式] 回放 V1 基线并运行差分测试...")
        print("=" * 50)
        run_test_cases_v2(baseline_path=baseline_path, report_path=report_path)
    else:
        base_no_ext = baseline_path.replace(".json", "")
        for page_index in range(20):
            old_path = baseline_path if page_index == 0 else f"{base_no_ext}_{page_index}.json"
            if os.path.exists(old_path):
                os.remove(old_path)
        print("=" * 50)
        print("[V1 模式] 运行基准录制...")
        print("=" * 50)
        run_test_cases_v1(K=K, output_path=baseline_path)
        # 自动生成递归 bug 汇总报告
        generate_bug_report(result_path=baseline_path)
