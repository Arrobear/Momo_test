from config import *
from typing import List, Dict
from stage_1_function import *

TORCH_PATH = Path("C:/Users/86184/Desktop/Papers/dl_lib/pytorch-2.5.1") # 修改为本地 PyTorch 源码根目录
YAML_PATH = TORCH_PATH / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
joern_bat_path = "C:/Users/86184/Desktop/joern-cli/joern.bat"

# =====================================================
# Joirn 交互式封装
# =====================================================
# joern 交互式 class 封装
class JoernShell:
    def __init__(self, joern_bat_path):
        """
        初始化 Joern shell
        joern_bat_path: joern.bat 的完整路径
        """
        self.process = subprocess.Popen(
            [joern_bat_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )

    def send_command(self, cmd):
        """
        向 Joern 发送命令，并读取 stdout 输出
        支持每条命令唯一 marker，避免上一次命令干扰
        """
        marker = f"__JOERN_CMD_DONE_{uuid.uuid4().hex}__"

        # 发送命令
        self.process.stdin.write(f"{cmd}\n")
        self.process.stdin.flush()

        # 发送 marker 打印
        self.process.stdin.write(f'println("{marker}")\n')
        self.process.stdin.flush()

        # 读取输出直到 marker 出现
        output_lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                break  # 进程结束
            if marker in line:
                break
            output_lines.append(line)
        return "".join(output_lines)
# 解析 Joern switch 输出为 Python 列表
def parse_joern_multiline(test_str):
    """
    解析多行 Joern 输出字符串，转换为 Python 列表形式
    """
    result = []
    for line in test_str.strip().splitlines():
        # 去掉开头的 'joern> ' 前缀
        line = line.strip()
        if line.startswith("joern>"):
            line = line[len("joern>"):].strip()
        if "->" not in line:
            continue
        var, cases_str = line.split("->", 1)
        cases = cases_str.split(",")
        for c in cases:
            if c.strip() == "default":
                result.append(f"{var} other")
            else:
                result.append(f"{var} == {c.strip()}")
    return result
# 解析 Joern Scala List 输出为 Python 列表
def parse_scala_list(scala_output: str):

    # 提取 List(...) 中的内容
    items = re.findall(r'"(.*?)"', scala_output)
    return items


# =====================================================
# Torch API 分类与 Guard 抽取
# =====================================================
# 判断 torch库中的 API 类型
def torch_api_classify(api_name: str) -> str:
    """
    改进版 PyTorch API 分类器
    """
    try:
        mod_name, attr_name = api_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr_name)
    except Exception:
        return "unknown"

    # 1️⃣ nn.functional → function
    if "torch.nn.functional" in mod_name:
        return "function"

    # 2️⃣ 类对象
    if inspect.isclass(obj):
        return "class"

    # 3️⃣ Tensor 方法
    if inspect.ismethoddescriptor(obj):
        return "method"

    # 4️⃣ Python 纯函数
    if inspect.isfunction(obj):
        return "function"

    # 5️⃣ 工厂函数
    factory_names = {"ones", "zeros", "arange", "randn", "empty", "full"}
    if inspect.isbuiltin(obj) and attr_name in factory_names:
        return "factory"

    # 6️⃣ C++ builtin (OpOverloadPacket / OpOverload)
    obj_type = str(type(obj))
    if (
        "torch._ops" in obj_type
        or hasattr(obj, "overloads")
        or hasattr(obj, "default")
        or hasattr(obj, "op")
        or type(obj).__name__ in {"OpOverload", "OpOverloadPacket"}
    ):
        # 尝试判断是否有 Python 层包装
        try:
            src = inspect.getsourcefile(obj)
            if src and "torch" in src and "site-packages" in src:
                return "function"
        except Exception:
            pass
        return "builtin"

    # 7️⃣ 常量
    if isinstance(obj, (int, float, str, bool, tuple, list, dict)):
        return "constant"

    return "unknown"


def torch_find_cpp_name(api_name: str) -> str:
    """
    从 native_functions.yaml 提取 C++ 实现函数名。
    支持 structured_delegate / autogen / CompositeAutograd 等。
    """
    func_target = api_name.split(".")[-1]

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        yaml_docs = yaml.safe_load(f)

    # -------- 辅助函数 --------
    def normalize_name(name: str) -> str:
        return name.strip().lower().replace(" ", "").replace(".", "_")

    def find_entry(target, fuzzy=False):
        target_norm = normalize_name(target)
        for entry in yaml_docs:
            func = entry.get("func")
            if not func:
                continue
            func_name_only = func.split("(", 1)[0].strip()
            func_norm = normalize_name(func_name_only)
            if not fuzzy:
                if func_norm == target_norm:
                    return entry
            else:
                if target_norm in func_norm:
                    return entry
        return None



    def extract_dispatch(dispatch):
        if not dispatch:
            return None
        # 允许键中包含多个后端（如 "CPU, CUDA"）
        for key, val in dispatch.items():
            key_clean = [k.strip().lower() for k in key.split(",")]
            if any(k in ["cpu", "cuda", "compositeexplicitautograd", "compositeimplicitautograd", "defaultbackend"] for k in key_clean):
                return val
        # fallback
        return list(dispatch.values())[0] if dispatch else None


    # -------- 判断 builtin --------
    is_builtin = "torch._C" in api_name or api_name.count(".") == 1

    # -------- 精确匹配 --------
    entry = find_entry(func_target, fuzzy=False)
    if entry:
        dispatch = entry.get("dispatch")
        structured_delegate = entry.get("structured_delegate")
        autogen = entry.get("autogen")

        # (1) 直接 dispatch
        impl = extract_dispatch(dispatch)
        if impl:
            print(f"[TRACE] {api_name} → {entry.get('func')} → {impl}")
            return impl

        # (2) structured_delegate
        if structured_delegate:
            delegate_entry = find_entry(structured_delegate, fuzzy=False)
            impl = extract_dispatch(delegate_entry.get("dispatch") if delegate_entry else None)
            if impl:
                print(f"[TRACE] {api_name} → delegate {structured_delegate} → {impl}")
                return impl

        # (3) autogen
        if autogen:
            autogen_entry = find_entry(autogen, fuzzy=False)
            impl = extract_dispatch(autogen_entry.get("dispatch") if autogen_entry else None)
            if impl:
                print(f"[TRACE] {api_name} → autogen {autogen} → {impl}")
                return impl

    # -------- 模糊匹配（仅 builtin）--------
    if is_builtin:
        entry = find_entry(func_target, fuzzy=True)
        impl = extract_dispatch(entry.get("dispatch") if entry else None)
        if impl:
            print(f"[TRACE] {api_name} → fuzzy match {entry.get('func')} → {impl}")
            return impl

    print(f"[WARN] 未找到 dispatch: {api_name}")
    return None

def torch_extract_cpp_guards(cpp_func_name: str) -> list:

    print(f"提取 C++ guards: {cpp_func_name}")
    cpp_guards = []
    # cpp_func_name = "testGuards"  # 临时测试用
    # -------- 3. 调用 joern 解析 C++ 层 guards --------
    joern = JoernShell(joern_bat_path)
    joern.send_command(f"open(\"{joern_project}\")")
    torch_cheack = joern.send_command(f"cpg.method.name(\"{cpp_func_name}\").call.name(\"TORCH_CHECK\").argument.order(1).code.l")
    print("torch_cheack_str:", torch_cheack)
    cheack_gtards = parse_scala_list(torch_cheack)

    torch_contorl = joern.send_command(f"cpg.method.name(\"{cpp_func_name}\").controlStructure.filterNot(_.controlStructureType == \"SWITCH\").condition.code.l")
    print("torch_cheack_str:", torch_contorl)
    contorl_gtards = parse_scala_list(torch_contorl)


    query_forhalf= f"cpg.method.name(\"{cpp_func_name}\")"
    query_backhalf = r""".ast.isControlStructure.filter(_.code.startsWith("switch")).foreach { sw => val cond = sw.code.split("\\(")(1).split("\\)")(0).trim; val cases = sw.astChildren.flatMap(_.astChildren).filter(n => n.code.startsWith("case") || n.code.startsWith("default")).toSeq.map(n => if (n.code.startsWith("case")) n.code.split(":")(0).replace("case","").trim else "default"); println(cond + "->" + cases.mkString(",")) }"""
    torch_switch = joern.send_command(query_forhalf + query_backhalf)
    switch_guards = parse_joern_multiline(torch_switch)

    joern.send_command(f"exit")   # 退出 Joern

    # -------- 4. 合并结果 --------
    cpp_guards.extend(cheack_gtards)
    cpp_guards.extend(contorl_gtards)  
    cpp_guards.extend(switch_guards)
    return cpp_guards

def torch_extract_python_guards(api_name: str) -> list:
    """
    抽取 Python 层 guards
    返回:
        python_guards: list[str]
    """
    python_guards = []

    # -------- 1. 找到 Python 对象 --------
    try:
        mod_name, attr_name = api_name.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[attr_name])
        py_obj = getattr(mod, attr_name)
    except Exception:
        return python_guards  # 找不到 API，返回空

    # -------- 2. 获取源码并构建 AST --------
    try:
        src = inspect.getsource(py_obj)
        tree = ast.parse(src)
    except (OSError, TypeError):
        return python_guards  # 一些 builtin 或 C++ API 没有源码
    except Exception:
        return python_guards

    # -------- 3. 遍历 AST 提取条件 --------
    class GuardVisitor(ast.NodeVisitor):
        def visit_If(self, node):
            try:
                cond = ast.unparse(node.test)
            except Exception:
                cond = "<complex_expr>"
            python_guards.append(cond)
            self.generic_visit(node)

        def visit_Assert(self, node):
            try:
                cond = ast.unparse(node.test)
            except Exception:
                cond = "<complex_expr>"
            python_guards.append(cond)
            self.generic_visit(node)

        def visit_Call(self, node):
            # torch._assert, _check_* 等函数调用
            if isinstance(node.func, ast.Name) and node.func.id in {"_assert", "_check"}:
                try:
                    cond = ast.unparse(node.args[0])
                except Exception:
                    cond = "<complex_expr>"
                python_guards.append(cond)
            self.generic_visit(node)

    GuardVisitor().visit(tree)

    return python_guards

def torch_extract_function_guards(api_name: str):
    """
    抽取 function 类型 API 的 guards
    
    返回字典:
        {
            "python_guards": [条件表达式列表],
            "cpp_guards": [条件表达式列表或文件定位信息]
        }
    
    参数:
        api_name: API 名称，例如 'torch.nn.functional.conv1d'
    """
    
    # -------- Python 层抽取 --------
    python_guards = torch_extract_python_guards(api_name)

    # -------- C++ 层 --------
    fun_cpp_name = torch_find_cpp_name(api_name)
    cpp_guards = torch_extract_cpp_guards(fun_cpp_name)

    return {
        "python_guards": python_guards,
        "cpp_guards": cpp_guards
    }

def torch_extract_builtin_guards(api_name: str):
    """
    抽取 builtin 函数的 guards（Python 层不可用，全部在 C++）
    
    返回字典:
        {
            "python_guards": [],  # builtin 没有 Python 层 guard
            "cpp_guards": [条件列表]
        }
    """
    python_guards = torch_extract_python_guards(api_name)  # Python 层为空
    fun_cpp_name = torch_find_cpp_name(api_name)
    #print("fun_cpp_name:", fun_cpp_name)
    cpp_guards = torch_extract_cpp_guards(fun_cpp_name)
    
    return {
        "python_guards": python_guards,
        "cpp_guards": cpp_guards
    }

def torch_extract_factory_guards(api_name: str):
    """
    抽取 factory 类型 API 的 guards（Python 层 + C++ 层）
    
    返回字典:
        {
            "python_guards": [条件列表],  # 如果 Python 层有检查
            "cpp_guards": [条件列表]
        }
    """
    # -------- Python 层抽取 --------
    python_guards = torch_extract_python_guards(api_name)
    # -------- C++ 层抽取 --------
    fun_cpp_name = torch_find_cpp_name(api_name)
    cpp_guards = torch_extract_cpp_guards(fun_cpp_name)

    return {
        "python_guards": python_guards,
        "cpp_guards": cpp_guards
    }

def torch_extract_class_guards(api_name: str):
    """
    抽取 class 类型 API 的 guards（Python 层 + C++ 层）
    递归追踪 forward 内部 helper（如 _conv_forward），保证 Python/C++ guard 可获取
    返回:
        {
            "python_guards": [...],
            "cpp_guards": [...]
        }
    """
    python_guards = []
    cpp_guards = []

    # -------- 1. 加载 class --------
    try:
        mod_name, cls_name = api_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls_obj = getattr(mod, cls_name, None)
    except Exception:
        cls_obj = None

    if cls_obj is None or not inspect.isclass(cls_obj):
        return {"python_guards": python_guards, "cpp_guards": cpp_guards}

    visited = set()

    # -------- 内部递归函数 --------
    def _analyze_method(method_name):
        nonlocal python_guards, cpp_guards

        if method_name in visited:
            return
        visited.add(method_name)

        py_func = getattr(cls_obj, method_name, None)
        if py_func is None:
            return

        try:
            src = inspect.getsource(py_func)
            src = textwrap.dedent(src)
            tree = ast.parse(src)
        except Exception:
            return

        # 1. 收集 if guards
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                try:
                    cond = ast.unparse(node.test)
                except Exception:
                    cond = ast.dump(node.test)
                python_guards.append(cond)

        # 2. 收集调用
        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        full_name = f"{node.func.value.id}.{node.func.attr}"
                    else:
                        full_name = node.func.attr
                    self.calls.append(full_name)
                elif isinstance(node.func, ast.Name):
                    self.calls.append(node.func.id)
                self.generic_visit(node)

        cv = CallVisitor()
        cv.visit(tree)

        for call in cv.calls:
            try:
                # class 内部 helper (self.xxx)
                if call.startswith("self."):
                    inner_name = call.split(".", 1)[1]
                    _analyze_method(inner_name)

                # functional / torch
                elif call.startswith("F.") or call.startswith("torch."):
                    from_module = "torch.nn.functional" if call.startswith("F.") else "torch"
                    func_name = call.split(".")[-1]
                    full_api_name = f"{from_module}.{func_name}"

                    try:
                        mod = importlib.import_module(from_module)
                        py_obj = getattr(mod, func_name, None)
                    except Exception:
                        py_obj = None

                    if py_obj is not None:
                        # -------- (1) 尝试解析 Python 源码 --------
                        try:
                            src_func = inspect.getsource(py_obj)
                            src_func = textwrap.dedent(src_func)
                            tree_func = ast.parse(src_func)

                            for node in ast.walk(tree_func):
                                if isinstance(node, ast.If):
                                    try:
                                        cond = ast.unparse(node.test)
                                    except Exception:
                                        cond = ast.dump(node.test)
                                    python_guards.append(cond)
                        except Exception:
                            pass

                        # -------- (2) unwrap boolean_dispatch --------
                        visited_py = set()
                        def unwrap(f):
                            if f in visited_py or f is None:
                                return f
                            visited_py.add(f)
                            for attr in ["if_true", "if_false"]:
                                if hasattr(f, attr):
                                    inner_f = getattr(f, attr)
                                    fun_cpp_name = torch_find_cpp_name(full_api_name)
                                    cpp_guards.extend(torch_extract_cpp_guards(fun_cpp_name))
                                    unwrap(inner_f)
                            return f
                        unwrap(py_obj)

                    # -------- (3) 最后调用 C++ guard 提取 --------
                    fun_cpp_name = torch_find_cpp_name(full_api_name)
                    cpp_guards.extend(torch_extract_cpp_guards(fun_cpp_name))

            except Exception:
                continue

    # -------- 入口: forward --------
    if hasattr(cls_obj, "forward"):
        _analyze_method("forward")

    return {"python_guards": python_guards, "cpp_guards": cpp_guards}

# print(torch_extract_cpp_guards("conv1d_symint"))
# print(torch_api_classify("torch.nn.functional.embedding"))



# =====================================================
# 规范化
# =====================================================
# ========== 一、参数过滤 ==========
def filter_guards_by_args(guards: list[str], api_name: str, keep_self: bool = False) -> list[str]:
    """
    仅保留与 API 参数相关的 guard。
    """
    arg_names = set(get_all_parameters(api_name) or [])
    if not keep_self and "self" in arg_names:
        arg_names.discard("self")

    if not arg_names:
        return guards

    arg_patterns = [re.compile(rf"\b{re.escape(arg)}\b") for arg in arg_names]

    def mentions_any_arg(expr: str) -> bool:
        if not isinstance(expr, str) or not expr.strip():
            return False
        s = re.sub(r"\s+", " ", expr).strip()
        return any(p.search(s) for p in arg_patterns)

    return [g for g in guards if mentions_any_arg(g)]

# ========== 二、表达式清理 ==========
def clean_expr(expr: str) -> str:
    """去除多余空格和括号"""
    expr = re.sub(r"\s+", " ", expr.strip())
    # 去掉最外层匹配括号
    while expr.startswith("(") and expr.endswith(")") and expr.count("(") == expr.count(")"):
        expr = expr[1:-1].strip()
    return expr

# ========== 三、类型推断 ==========

def infer_guard_type(expr: str) -> str:
    """
    根据 guard 内容推断其逻辑类型。
    """
    if "dtype" in expr:
        return "dtype_check"
    elif "device" in expr:
        return "device_check"
    elif "shape" in expr or "size" in expr:
        return "shape_check"
    elif "None" in expr:
        return "existence_check"
    elif re.search(r"(>|<|>=|<=|==|!=)", expr):
        return "range_check"
    elif re.search(r"in |not in ", expr):
        return "membership_check"
    else:
        return "boolean"
    
# ========== 四、guard结构化 ==========
def normalize_guard(expr: str, src: str) -> dict:
    """
    将 guard 标准化为结构化形式。
    """
    expr = clean_expr(expr)
    pattern = re.compile(r"(==|!=|>=|<=|>|<| in | not in | is | is not)")
    m = pattern.search(expr)
    if m:
        op = m.group(1).strip()
        lhs = expr[:m.start()].strip()
        rhs = expr[m.end():].strip()
    else:
        lhs, rhs, op = expr, "", ""
    return {
        "lhs": lhs,
        "op": op,
        "rhs": rhs,
        "expr": expr,
        "type": infer_guard_type(expr),
        "src": src
    }

# ========== 五、规范化主函数 ==========
def normalize_guards_stage(raw_guards: dict, api_name: str) -> dict:
    """
    规范化 Python + C++ guards，生成路径枚举友好格式。
    """
    result = {"python": [], "cpp": []}

    for src in ["python", "cpp"]:
        guards = raw_guards.get(f"{src}_guards", [])
        guards = list({g.strip() for g in guards if g.strip()})  # 去重
        guards = filter_guards_by_args(guards, api_name)

        normalized = [normalize_guard(g, src) for g in guards]
        result[src] = normalized

    # 合并为路径枚举友好格式（含来源）
    for_path_enum = []
    for src, guards in result.items():
        for g in guards:
            for_path_enum.append({
                "expr": g["expr"],
                "src": src,
                "type": g["type"]
            })

    return {
        "normalized_guards": result,
        "for_path_enumeration": for_path_enum
    }


# =====================================================
# 批量提取并规范化 guards
# =====================================================

def generate_normalized_guards(api_names: list[str], output_path: str):
    """
    批量提取并规范化 guards。
    每处理完一个 API 就立即写入输出 JSON 文件（防止内存过大 / 崩溃丢失）。
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 如果文件已存在，则加载旧数据，否则初始化空字典
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    # 遍历 API
    for api in api_names:
        if api in all_results:
            print(f"⏩ Skipping already processed API: {api}")
            continue

        print(f"\n[+] Processing API: {api}")

        try:
            # 1️⃣ 分类
            api_type = torch_api_classify(api)
            print(f"    ↳ Type: {api_type}")

            # 2️⃣ 抽取 guards
            if api_type == "function":
                raw_guards = torch_extract_function_guards(api)
            elif api_type == "class":
                raw_guards = torch_extract_class_guards(api)
            elif api_type == "builtin":
                raw_guards = torch_extract_builtin_guards(api)
            elif api_type == "factory":
                raw_guards = torch_extract_factory_guards(api)
            else:
                print(f"     Unknown type ({api_type}), skipping.")
                continue

            # 3️⃣ 规范化
            normalized = normalize_guards_stage(raw_guards, api)

            # 4️⃣ 写入内存结构
            all_results[api] = {
                "type": api_type,
                "normalized_guards": normalized["normalized_guards"],
                "for_path_enumeration": normalized["for_path_enumeration"]
            }

            # 5️⃣ 立刻保存到文件（覆盖写入）
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            print(f"     Saved progress for {api}")

        except Exception as e:
            print(f"     Error processing {api}: {e}")
            continue

    print(f"\n✅ All APIs processed and saved to: {output_file}")


# =====================================================
# 路径枚举
# =====================================================

# Step 1️⃣ Python 层路径枚举
def enumerate_python_paths(api_name: str, api_data: dict):
    """
    根据 Python 层 guards 生成路径条件
    每个 guard 默认包含 True/False 分支
    """
    guards_py = api_data.get("for_path_enumeration", [])
    calls_cpp = api_data.get("calls_cpp", False)

    if not guards_py:
        return []

    expanded = []
    for g in guards_py:
        expr = g[0] if isinstance(g, list) else g
        expanded.append([expr, f"not ({expr})"])

    path_conditions = []
    for combo in itertools.product(*expanded):
        cond = " and ".join(combo)
        path_conditions.append({
            "id": f"{api_name}_P{len(path_conditions)+1}",
            "expr": cond,
            "calls_cpp": calls_cpp,
            "src": "python"
        })

    return path_conditions


# Step 2️⃣ C++ 层路径提取
def enumerate_cpp_paths(api_name: str, api_data: dict):
    """
    生成 C++ 层路径条件（guards 已经由前一阶段提取）
    """
    guards_cpp = api_data.get("cpp_guards", [])
    if not guards_cpp:
        return []

    cpp_paths = []
    for idx, cond in enumerate(guards_cpp, 1):
        cpp_paths.append({
            "id": f"{api_name}_C{idx}",
            "expr": cond,
            "src": "cpp"
        })
    return cpp_paths


# Step 3️⃣ 路径合并与参数空间生成

def combine_paths(api_name: str, py_paths: list, cpp_paths: list):
    """
    将 Python 路径与 C++ 路径组合成参数空间
    """
    result = []
    for py_p in py_paths:
        if py_p["calls_cpp"] and cpp_paths:
            # 若路径触发 C++ 调用，则做笛卡尔积
            for cpp_p in cpp_paths:
                combined_cond = f"({py_p['expr']}) and ({cpp_p['expr']})"
                result.append({
                    "id": f"{py_p['id']}_C{cpp_p['id']}",
                    "expr": combined_cond,
                    "src": "py+cpp"
                })
        else:
            # 否则仅保留 Python 路径
            result.append(py_p)
    return result

# 整体入口

def generate_parameter_spaces(normalized_path: str, output_path: str):
    """
    读取已规范化 guards 文件，生成参数空间（路径枚举结果）
    """
    normalized_path = Path(normalized_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(normalized_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = {}

    for api_name, api_data in data.items():
        print(f"[+] Enumerating parameter spaces for {api_name}")
        try:
            # Step 1️⃣ Python
            py_paths = enumerate_python_paths(api_name, api_data)

            # Step 2️⃣ C++
            cpp_paths = enumerate_cpp_paths(api_name, api_data)

            # Step 3️⃣ 合并
            combined = combine_paths(api_name, py_paths, cpp_paths)

            all_results[api_name] = {
                "spaces": combined,
                "summary": {
                    "python_paths": len(py_paths),
                    "cpp_paths": len(cpp_paths),
                    "total_combined": len(combined)
                }
            }
        except Exception as e:
            print(f"  ❌ Error processing {api_name}: {e}")

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 路径枚举阶段完成，结果已保存到：{output_path}")


# ========================
# Example Run
# ========================
if __name__ == "__main__":
    generate_parameter_spaces(
        normalized_path="output/all_api_guards.json",
        output_path="output/all_api_paths.json"
    )





# if __name__ == "__main__":
    
#     # api_names= read_file(f"./documentation/{lib_name}_APIdef.txt")

#     # generate_normalized_guards(api_names, "normalized_guards.json")

#     print(torch_extract_function_guards("torch.nn.functional.conv1d"))