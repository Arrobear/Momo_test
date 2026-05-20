from config import *
from stage_1_function import *
from generate_prompt import *
'''
存储整个方法中的小步骤 

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''
def generate_api_conditions(api_names):
    # 初始化 DeepSeek 客户端
    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )

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

        # 生成prompt
        prompt_1 = generate_prompt_1(fun_string, api_def, api_doc)
        # print("_________________________________________________________________________________________________________")
        # print(prompt_1)
        
        # 调用线上 API 替代本地推理
        try:
            response = client.chat.completions.create(
                model="gpt-5.5",  # 或者使用 deepseek-reasoner
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_1},
                ],
                stream=False
            )
            outputs_text = response.choices[0].message.content
        except Exception as e:
            print(f"[API 错误] {e}")
            if i >= len(api_names): break
            continue

        print("_________________________________________________________________________________________________________")
        print(outputs_text)
        
        # handle_output 需要根据线上输出微调（API 不会带入 prompt 本身，仅输出结果）
        # 传递 model_path 可能是为了在 handle_output 中做逻辑判断，予以保留
        api_conditions = extract_clean_json(outputs_text)
        
        print("_________________________________________________________________________________________________________")
        print(api_conditions)
        
        # 存储至json
        path = root_path + f'/haoyahui/documentation/conditions/{lib_name}_conditions.json'

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

        json_path = root_path + f'/haoyahui/documentation/conditions/{lib_name}_conditions.json'
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
            path = root_path + f'/haoyahui/documentation/arg_combinations/{lib_name}_combinations_{j}.json'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if os.path.exists(path):
                if is_file_too_large(path, max_size_mb=10):
                    j += 1
                    path = root_path + f'/haoyahui/documentation/arg_combinations/{lib_name}_combinations_{j}.json'
            
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
    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )

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

        arg_combinations, j = get_all_combinations_from_json(function_name, j)
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
            try:
                response = client.chat.completions.create(
                    model="gpt-5.5",
                    messages=[
                        {"role": "system", "content": "You are a professional software testing assistant."},
                        {"role": "user", "content": prompt_2},
                    ],
                    stream=False
                )
                outputs_text = response.choices[0].message.content
                print(outputs_text)
            except Exception as e:
                print(f"[API 错误] 函数 {function_name} 在请求时发生异常: {e}")
                outputs_text = "" # 或者根据业务逻辑选择 continue

            # 处理输出并判断
            # 注意：API 返回的 outputs_text 不包含 prompt，handle_output 逻辑可能需要适配
            #error_tag = handle_output(outputs_text, model_path)
            # print(error_tag)
            if 'False' in outputs_text:
                error_combinations.append(arg_combination)
            
            n += 1
            print(f"API: {function_name} | 进度：{n}/{len(arg_combinations)}")
        # --------------------------------
        path = root_path + f'/haoyahui/documentation/error_combinations/error_{lib_name}_combinations.json'
        # path = f'/tmp/Momo_test/error_combinations/error_{lib_name}_combinations.json'
        append_filtered_combinations_to_json(path, function_name, error_combinations)

        if i >= len(api_names):
            break



# 剪枝后组合

def generate_api_boundary(api_names):
    # 初始化 DeepSeek 客户端
    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )

    # 移除 if lib_name == "torch" 判断，直接进入通用流程
    j = 0
    path = root_path + f'/haoyahui/documentation/arg_boundary/cut_{lib_name}_boundary_{j}.json'
    length_api_names = len(api_names)
    i = 0

    while i < length_api_names:
        api_inputs = []
        api_name = filter_samenames(i, api_names[i], api_names)
        
        # 读取相关的 JSON 配置
        arg_combinations = read_json_api(api_name=api_name, file_path=f"../documentation/arg_combinations/", read_mode="cut_combination")
        conditions = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
        arg_spaces = read_json_api(api_name=api_names[i], file_path=f"../documentation/arg_space/", read_mode="arg_space")

        if arg_spaces is None:
            add_log(root_path + f"/haoyahui/Momo_test/", api_name)
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
                print(f"API进度: {i+1}/{length_api_names} | {api_name}")


                path_type = arg_space["path_type"]
                prompt = generate_prompt_3(api_name, comb, arg_space, conditions["Parameter type"])

                # --- 调用线上 API ---
                try:
                    response = client.chat.completions.create(
                        model="gpt-5.5",
                        messages=[
                            {"role": "system", "content": "You are a specialized AI for API boundary analysis and software testing."},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False
                    )
                    outputs_text = response.choices[0].message.content
                except Exception as e:
                    print(f"[API 错误] 无法获取 {api_name} 的响应: {e}")
                    outputs_text = ""

                # 解析输出
                api_boundary = extract_clean_json(outputs_text)

                new_api_input_boundary = {"path_type": path_type, "api_input": api_boundary}
                api_inputs.append(new_api_input_boundary)

        # 存储至 JSON
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path) and is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = root_path + f'/haoyahui/documentation/arg_boundary/{lib_name}_boundary_{j}.json'
        
        save_api_inputs(api_name, api_inputs, path)
        
        i += 1

    return



#------------------------------------
# 生成默认输入
#------------------------------------
def generate_default_inputs(api_names):
    # 初始化 DeepSeek 客户端
    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )

    # 重新从文件读取最新的 API 列表
    api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")

    j = 0
    path = root_path + f'/haoyahui/documentation/api_input/{lib_name}_default_inputs_{j}.json'
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
        try:
            response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "You are a specialized AI assistant for generating default API inputs and test cases."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            outputs_text = response.choices[0].message.content
        except Exception as e:
            print(f"[API 错误] 函数 {api_name} 请求失败: {e}")
            outputs_text = ""

        # 使用之前修改好的 extract_clean_json 抽取 JSON
        api_default_input = extract_clean_json(outputs_text)
                
        # 存储至 json
        # 检查文件大小，必要时切换文件编号
        if os.path.exists(path) and is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = root_path + f'/haoyahui/documentation/api_input/{lib_name}_default_inputs_{j}.json'
        
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
        path = root_path + f'/haoyahui/documentation/api_input/{lib_name}_inputs_{j}.json'
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
                path = root_path + f'/haoyahui/documentation/api_input/{lib_name}_input_{j}.json'
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
    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )

    j = 0
    path = root_path + f'/haoyahui/documentation/api_input/{lib_name}_inputs_{j}.json'
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
            try:
                response = client.chat.completions.create(
                    model="gpt-5.5",
                    messages=[
                        {"role": "system", "content": "You are a specialized AI assistant for generating API test inputs and test cases."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                    stream=False
                )
                outputs_text = response.choices[0].message.content
            except Exception as e:
                print(f"[API 错误] 函数 {api_name} 参数 {key} 请求失败: {e}")
                outputs_text = ""

            arg_input = extract_clean_list(outputs_text)
            # 为每个参数打上类型标签，区分 code 字符串和 literal 字符串
            param_type = "code" if _is_code_type(value) else "literal"
            api_inputs_candidate[key] = {"type": param_type, "values": arg_input}

        # 存储至 json
        if is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = root_path + f'/haoyahui/documentation/api_input/{lib_name}_inputs_{j}.json'
        save_api_inputs(api_name, api_inputs_candidate, path)
        print(f"已完成{api_name}的API输入生成, 进度 {i+1}/{length_api_names}")
        i += 1
    return

#------------------------------------
# 生成测试案例model
#------------------------------------
def generate_test_cases(api_names):

    client = OpenAI(
        api_key = API_KEY,
        base_url = BASE_URL
    )

    # 读取 API 定义
    with open(f"../documentation/lib_api/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")

    j = 0
    path = f"{root_path}/haoyahui/documentation/test_cases/{lib_name}_case_{j}.json"

    for i in range(len(api_names)):
        api_name = api_names[i]
        function_name = filter_samenames(i, api_name, api_names)
        api_def = api_defs[i]

        api_doc = get_doc(function_name)
        prompt_6 = generate_prompt_6(api_name, api_def, api_doc)

        try:
            response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "You are a specialized AI assistant for generating API test inputs and test cases."},
                    {"role": "user", "content": prompt_6},
                ],
                temperature=0.0,
                top_p=1.0,
                seed=42,
                stream=False
            )
            outputs_text = response.choices[0].message.content
        except Exception as e:
            print(f"API请求失败 [{api_name}]: {e}")
            continue

        case = outputs_text

        if is_file_too_large(path, max_size_mb=1000):
            j += 1
            path = f"{root_path}/haoyahui/documentation/test_cases/{lib_name}_case_{j}.json"

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
        exec(code_str, globals(), local_namespace)
        return local_namespace.get("run_api")
    except Exception as e:
        print(f"解析 {api_name} 的 case 代码失败: {e}")
        return None


def _eval_param_by_type(param_value, param_type):
    """根据类型标签决定是否 eval：code 类型 eval，literal 类型原样返回"""
    if param_type == "code" and isinstance(param_value, str):
        try:
            return eval(param_value)
        except Exception:
            return param_value
    return param_value


def run_test_cases_v1(K=100, output_path=None):
    """
    运行测试案例框架（差分测试 V1 基准录制版）
    :param K: 每个 API 组装并执行的测试用例数量
    :param output_path: 基线输出路径，默认 results/{lib_name}_v1_baseline.json
    """
    api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")
    if output_path is None:
        output_path = root_path + f'/haoyahui/documentation/results/{lib_name}_v1_baseline.json'
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

        # 2. 组装并执行 K 次测试用例
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

            result_entry = {
                "测试输入": serialized_input,
                "函数返回结果": None,
                "函数运行状态": "pending"
            }

            # --- 参数解析 ---
            evaluated_item = {}
            eval_success = True
            for param_name, entry in assembled.items():
                try:
                    evaluated_item[param_name] = _eval_param_by_type(entry["value"], entry["type"])
                except Exception as e:
                    eval_success = False
                    result_entry["函数运行状态"] = "error"
                    result_entry["函数返回结果"] = f"参数解析错误 ({param_name}): {str(e)}"
                    break

            if not eval_success:
                api_run_results.append(result_entry)
                continue

            # --- 运行 API ---
            try:
                output = run_api(**evaluated_item)
                result_entry["函数运行状态"] = "success"
                result_entry["函数返回结果"] = safe_serialize(output)
            except Exception as e:
                result_entry["函数运行状态"] = "error"
                result_entry["函数返回结果"] = f"{type(e).__name__}: {str(e)}"

            api_run_results.append(result_entry)

        # 3. 文件切分与保存逻辑
        j, path = save_and_paginate(api_name, api_run_results, path, root_path, lib_name, j, page_pattern)
        print(f"已完成 {api_name} 的API测试 (执行 {K} 次), 进度 {i+1}/{len(api_names)}")



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
        baseline_path = root_path + f'/haoyahui/documentation/results/{lib_name}_v1_baseline.json'
    if report_path is None:
        report_path = root_path + f'/haoyahui/documentation/results/{lib_name}_diff_report.json'

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

        # 3. 遍历 V1 记录的每个用例
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
                except Exception:
                    eval_success = False
                    break

            if not eval_success:
                continue

            # --- 运行 V2 API ---
            v2_result = None
            v2_status = "pending"
            try:
                output = run_api(**evaluated_item)
                v2_result = safe_serialize(output)
                v2_status = "success"
            except Exception as e:
                v2_result = f"{type(e).__name__}: {str(e)}"
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
    baseline_path = root_path + f'/haoyahui/documentation/results/{lib_name}_v1_baseline.json'
    report_path = root_path + f'/haoyahui/documentation/results/{lib_name}_diff_report.json'

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