from config import *
from typing import List, Dict
from stage_1_function import *
'''
用于在 PyTorch 库中抽取 API 的 Guard 条件。


'''


TORCH_PATH = Path("C:/Users/86184/Desktop/Papers/dl_lib/pytorch-2.5.1") # 修改为本地 PyTorch 源码根目录
YAML_PATH = TORCH_PATH / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"


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
            encoding="gbk",
            errors="replace",
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
    强化版 PyTorch API 分类器。
    支持：Python 层 + C++ 层 + YAML fallback。
    """
    try:
        mod_name, attr_name = api_name.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr_name)
    except Exception:
        obj = None

    # 1️⃣ nn.functional 明确为 function
    if "torch.nn.functional" in api_name:
        return "function"

    # 2️⃣ 类
    if obj and inspect.isclass(obj):
        return "class"

    # 3️⃣ 工厂函数
    factory_names = {"ones", "zeros", "empty", "full", "arange", "randn", "rand", "eye", "linspace"}
    if attr_name in factory_names:
        return "factory"

    # 4️⃣ 普通 Python 函数
    if obj and inspect.isfunction(obj):
        return "function"

    # 5️⃣ Tensor 实例方法
    if obj and inspect.ismethoddescriptor(obj):
        return "method"

    # 6️⃣ 尝试识别内建函数 / OpOverload
    obj_type = str(type(obj))
    if obj and ("OpOverload" in obj_type or "OpOverloadPacket" in obj_type):
        try:
            src = inspect.getsourcefile(obj)
            if src and "torch" in src and not src.endswith(".pyd"):
                return "function"
        except Exception:
            return "builtin"
        return "builtin"

    # 7️⃣ fallback：从 native_functions.yaml 查找
    try:
        import yaml
        func_target = attr_name.split(".")[-1]
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            yaml_docs = yaml.safe_load(f)
        for entry in yaml_docs:
            func = entry.get("func", "")
            if func.startswith(func_target + "("):
                return "builtin"
    except Exception:
        pass

    # 8️⃣ torch._C / _ops 注册的直接算子
    if "torch._C" in api_name or "torch._ops" in api_name:
        return "builtin"

    # 9️⃣ 全部失败，返回 unknown
    return "unknown"

def torch_find_cpp_name(api_name: str) -> str:
    """
    从 native_functions.yaml 提取 C++ 实现函数名。
    支持 structured_delegate / autogen / CompositeAutograd 等。
    """
    func_target = api_name.split(".")[-1]
    # print(func_target)
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        yaml_docs = yaml.safe_load(f)

    # -------- 辅助函数 --------
    def normalize_name(name: str) -> str:
        return name.strip().lower().replace(" ", "").replace(".", "_")

    def find_entry(target):
        target_norm = normalize_name(target)

        matched_entries = []
        started = False  # 是否已经开始进入连续匹配阶段

        for entry in yaml_docs:
            func = entry.get("func")
            if not func:
                # 如果还没开始匹配，继续跳过
                if not started:
                    continue
                else:
                    # 已经开始匹配了，但现在没 func，算不匹配 → 停止
                    break

            func_name_only = func.split("(", 1)[0].strip()
            func_norm = normalize_name(func_name_only)

            if target_norm in func_norm:
                matched_entries.append(entry)
                started = True  # 进入连续匹配模式
            else:
                # 如果已经进入匹配阶段，但现在不匹配 → 停止
                if started:
                    break
                # 如果还没开始匹配 → 忽略继续找第一个
                continue

        # 如果没有任何匹配项
        if not matched_entries:
            return None

        # 优先挑出真正实现的 entry（CPU/CUDA kernel）
        for entry in matched_entries:
            dispatch = entry.get("dispatch", {})
            if "CPU" in dispatch or "CUDA" in dispatch:
                return entry

        # 如果没有 kernel 实现，fallback 返回第一个匹配项
        return matched_entries[0]


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

    #cpp_name = None
    # -------- 判断 builtin --------
    is_builtin = "torch._C" in api_name or api_name.count(".") == 1

    # -------- 精确匹配 --------
    entry = find_entry(func_target)
    if entry:
        dispatch = entry.get("dispatch")
        structured_delegate = entry.get("structured_delegate")
        autogen = entry.get("autogen")

        # (1) 直接 dispatch
        impl = extract_dispatch(dispatch)
        if impl:
           # print(f"[TRACE] {api_name} → {entry.get('func')} → {impl}")
            return impl

        # (2) structured_delegate
        if structured_delegate:
            delegate_entry = find_entry(structured_delegate)
            impl = extract_dispatch(delegate_entry.get("dispatch") if delegate_entry else None)
            if impl:
                # print(f"[TRACE] {api_name} → delegate {structured_delegate} → {impl}")
                return impl

        # (3) autogen
        if autogen:
            autogen_entry = find_entry(autogen)
            impl = extract_dispatch(autogen_entry.get("dispatch") if autogen_entry else None)
            if impl:
                # print(f"[TRACE] {api_name} → autogen {autogen} → {impl}")
                return impl

    # -------- 模糊匹配（仅 builtin）--------
    if is_builtin:
        entry = find_entry(func_target)
        impl = extract_dispatch(entry.get("dispatch") if entry else None)
        if impl:
            # print(f"[TRACE] {api_name} → fuzzy match {entry.get('func')} → {impl}")
            return impl

    # print(f"[WARN] 未找到 dispatch: {api_name}")
    return None


def torch_extract_cpp_guards(cpp_func_name: str) -> list:
    """
    使用 Joern 从 C++ 函数中提取 guard 条件，并自动生成正/反路径。
    包含：
        - TORCH_CHECK(cond)
        - if(cond)
        - for(cond)
        - while(cond)
        - switch(var)
    返回：
        cpp_guards: list[str]
    """
    # print(f"提取 C++ guards: {cpp_func_name}")
    cpp_guards = []

    joern = JoernShell(joern_bat_path)
    joern.send_command(f"open(\"{joern_project}\")")

    # 1️⃣ TORCH_CHECK 条件
    torch_check_str = joern.send_command(
        f"cpg.method.name(\"{cpp_func_name}\").call.name(\"TORCH_CHECK\").argument.order(1).code.l"
    )
    torch_checks = parse_scala_list(torch_check_str)
    for cond in torch_checks:
        cpp_guards.append(cond)
        cpp_guards.append(f"not ({cond})")

    # 2️⃣ if 条件
    if_str = joern.send_command(
        f"cpg.method.name(\"{cpp_func_name}\").controlStructure.controlStructureType(\"IF\").condition.code.l"
    )
    if_conds = parse_scala_list(if_str)
    for cond in if_conds:
        body_code = joern.send_command(
            f'cpg.method.name(\"{cpp_func_name}\").controlStructure.condition.code(\"{cond}\").astChildren.code.l'
        )
        if any(keyword in body_code for keyword in ["TORCH_CHECK", "throw", "return"]):
            cpp_guards.append(f"not ({cond})")
        else:
            cpp_guards.append(cond)
            cpp_guards.append(f"not ({cond})")

    # 3️⃣ for 条件
    for_str = joern.send_command(
        f"cpg.method.name(\"{cpp_func_name}\").controlStructure.controlStructureType(\"FOR\").condition.code.l"
    )
    for_conds = parse_scala_list(for_str)
    for cond in for_conds:
        cpp_guards.append(cond)
        cpp_guards.append(f"not ({cond})")

    # 4️⃣ while 条件
    while_str = joern.send_command(
        f"cpg.method.name(\"{cpp_func_name}\").controlStructure.controlStructureType(\"WHILE\").condition.code.l"
    )
    while_conds = parse_scala_list(while_str)
    for cond in while_conds:
        cpp_guards.append(cond)
        cpp_guards.append(f"not ({cond})")

    # 5️⃣ switch 条件
    query_forhalf = f"cpg.method.name(\"{cpp_func_name}\")"
    query_backhalf = r""".ast.isControlStructure.filter(_.code.startsWith("switch")).foreach { sw => 
        val cond = sw.code.split("\\(")(1).split("\\)")(0).trim
        val cases = sw.astChildren.flatMap(_.astChildren)
            .filter(n => n.code.startsWith("case") || n.code.startsWith("default"))
            .toSeq.map(n => if (n.code.startsWith("case")) 
                n.code.split(":")(0).replace("case","").trim else "default")
        println(cond + "->" + cases.mkString(","))
    }"""
    torch_switch = joern.send_command(query_forhalf + query_backhalf)
    switch_guards = parse_joern_multiline(torch_switch)
    cpp_guards.extend(switch_guards)

    # 6️⃣ 清理与去重 + 逻辑标准化
    def normalize_negation(expr: str) -> str:
        """将 !(x >= 0) → not (x >= 0)，并去除双重否定"""
        expr = expr.strip()
        expr = re.sub(r'!\s*\(', 'not (', expr)
        expr = re.sub(r'not\s*\(\s*not\s*\((.*?)\)\s*\)', r'\1', expr)
        expr = re.sub(r'\s+', ' ', expr.strip())
        return expr

    cpp_guards = [normalize_negation(g) for g in cpp_guards if g.strip()]
    cpp_guards = list({g.strip() for g in cpp_guards if g.strip()})


    joern.send_command("exit")
    # print(f"[CPP GUARDS] Extracted {len(cpp_guards)} guards from {cpp_func_name}")
    return cpp_guards

def torch_extract_python_guards(api_name: str) -> list:
    """
    抽取 Python 层 guards（正/反路径均提取）
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
    except (OSError, TypeError, SyntaxError):
        return python_guards

    # -------- 3. 遍历 AST 提取条件（增强版）--------
    class GuardVisitor(ast.NodeVisitor):
        def visit_If(self, node):
            try:
                cond = ast.unparse(node.test)
            except Exception:
                cond = "<complex_expr>"

            # 判断是否含 raise
            has_raise = any(isinstance(n, ast.Raise) for n in node.body)
            has_else_raise = any(isinstance(n, ast.Raise) for n in node.orelse)

            if has_raise:
                # if cond: raise -> feasible path 是 not(cond)
                python_guards.append(f"not ({cond})")
            elif has_else_raise:
                # else: raise -> feasible path 是 cond
                python_guards.append(cond)
            else:
                # 一般分支 -> 保留 cond 和 not(cond)
                python_guards.append(cond)
                python_guards.append(f"not ({cond})")

            # 继续递归遍历
            self.generic_visit(node)

        def visit_Assert(self, node):
            try:
                cond = ast.unparse(node.test)
            except Exception:
                cond = "<complex_expr>"
            # assert cond -> feasible path 是 cond
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

    # 去重与清理
    python_guards = list({g.strip() for g in python_guards if g.strip()})
    return python_guards

def torch_extract_function_guards(api_name: str):
    """
    抽取 function 类型 API 的 guards。
    包含 Python 层 + C++ 层。
    """
    python_guards = torch_extract_python_guards(api_name)

    cpp_guards = []
    fun_cpp_name = torch_find_cpp_name(api_name)
    if fun_cpp_name:
        try:
            cpp_guards = torch_extract_cpp_guards(fun_cpp_name)
        except Exception as e:
            print(f"[WARN] Failed to extract C++ guards for {api_name}: {e}")
    #else:
        # print(f"[WARN] No C++ mapping found for function API: {api_name}")

    return {
        "python_guards": python_guards,
        "cpp_guards": cpp_guards
    }

def torch_extract_builtin_guards(api_name: str):
    """
    抽取 builtin 类型 API 的 guards。
    优化：同时尝试 Python 层提取（若失败或为空则忽略），
    并始终提取 C++ 层 TORCH_CHECK / 控制语句。
    """
    python_guards = []
    cpp_guards = []

    # 🧩 尝试 Python 层提取（某些 builtin 实际有包装）
    try:
        python_guards = torch_extract_python_guards(api_name)
    except Exception as e:
        #print(f"[WARN] Python guard extraction failed for builtin {api_name}: {e}")
        python_guards = []

    # 🧩 提取 C++ 层
    try:
        fun_cpp_name = torch_find_cpp_name(api_name)
        if fun_cpp_name:
            cpp_guards = torch_extract_cpp_guards(fun_cpp_name)
        # else:
        #     print(f"[WARN] No C++ mapping found for builtin API: {api_name}")
    except Exception as e:
        # print(f"[WARN] C++ guard extraction failed for builtin {api_name}: {e}")
        cpp_guards = []

    return {
        "python_guards": python_guards,
        "cpp_guards": cpp_guards
    }

def torch_extract_factory_guards(api_name: str):
    """
    抽取 factory 类型 API 的 guards（如 torch.zeros / torch.arange）。
    一般无复杂 Python 逻辑，但可能有参数检查。
    """
    python_guards = torch_extract_python_guards(api_name)

    cpp_guards = []
    fun_cpp_name = torch_find_cpp_name(api_name)
    if fun_cpp_name:
        try:
            cpp_guards = torch_extract_cpp_guards(fun_cpp_name)
        except Exception as e:
            print(f"[WARN] Failed to extract C++ guards for factory {api_name}: {e}")
    else:
        print(f"[WARN] No C++ mapping found for factory API: {api_name}")

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

def torch_extract_unknown_guards(api_name: str):
    """
    对 unknown 类型也尽力而为：
    - 先尝试 Python 层 guard 提取（失败忽略）；
    - 再尝试通过 YAML 映射到 C++ 实现并提取 C++ guards（失败忽略）。
    """
    python_guards = []
    cpp_guards = []


    # Python 层（尽力而为）
    try:
        python_guards = torch_extract_python_guards(api_name) or []
    except Exception as e:
        #print(f"[WARN] Python guard extraction failed for unknown {api_name}: {e}")
        python_guards = []


    # C++ 层（尝试找到对应实现）
    try:
        fun_cpp_name = torch_find_cpp_name(api_name)
        if fun_cpp_name:
            try:
                cpp_guards = torch_extract_cpp_guards(fun_cpp_name) or []
            except Exception as e:
                print(f"[WARN] C++ guard extraction failed for unknown {api_name}: {e}")
        else:
            print(f"[WARN] No C++ mapping found for unknown API: {api_name}")
    except Exception as e:
        print(f"[WARN] torch_find_cpp_name failed for unknown {api_name}: {e}")

    return {"python_guards": python_guards, "cpp_guards": cpp_guards}

def torch_extract_guards(api_name: str):
    """
    统一调度函数，根据 API 类型自动调用对应的 guard 提取逻辑。
    对 unknown 类型：尝试 Python + C++ 双路径提取，失败则忽略。
    """
    api_type = torch_api_classify(api_name)
    #print(f"[INFO] Extracting guards for {api_name} (type: {api_type})")


    if api_type == "function":
        return torch_extract_function_guards(api_name)
    elif api_type == "builtin":
        return torch_extract_builtin_guards(api_name)
    elif api_type == "factory":
        return torch_extract_factory_guards(api_name)
    elif api_type == "class":
        return torch_extract_class_guards(api_name)
    else:
    # unknown → 也尝试两层提取
        return torch_extract_unknown_guards(api_name)

# print(torch_extract_cpp_guards("conv1d_symint"))
# print(torch_api_classify("torch.nn.functional.embedding"))

# =====================================================
# Guard 规范化阶段
# =====================================================

def filter_guards_by_args(guards: list[str], api_name: str, keep_self: bool = False) -> list[str]:
    """
    改进版（更宽松匹配逻辑）：
      - 仅要 guard 中含有任意参数名（如 input, bias）即保留；
      - 同时保留 dtype/device/scalar_type/isComplexType 等关键检查；
      - 不使用复杂正则，直接字符串包含判断；
      - 允许参数名出现在任意位置（例如 input_, self.input, at::isComplexType(input.scalar_type())）。
    """
    try:
        arg_names = list(get_all_parameters(api_name) or [])
    except Exception:
        arg_names = []

    if not keep_self and "self" in arg_names:
        arg_names.remove("self")

    if not arg_names:
        return guards  # 若无法识别参数，直接保留所有

    filtered = []
    for g in guards:
        if not isinstance(g, str) or not g.strip():
            continue

        # 宽松匹配：guard 中包含任意参数名即可
        keep = any(arg in g for arg in arg_names)

        # 保留 dtype/device/scalar_type/isComplexType 等关键类型检查
        if not keep and any(k in g for k in ["dtype", "device", "scalar_type", "isComplexType", "shape"]):
            keep = True

        if keep:
            filtered.append(g)

    # 去重保序
    return list(dict.fromkeys(filtered))

def clean_expr(expr: str) -> str:
    """
    清理表达式：
    - 去除多余空格和括号
    - 标准化逻辑符号（and→&&, or→||）
    - 展平多层 not()
    """
    if not isinstance(expr, str):
        return expr

    expr = expr.strip()
    expr = re.sub(r"\s+", " ", expr)
    expr = expr.replace(" and ", " && ").replace(" or ", " || ")

    # 去掉多余括号
    while expr.startswith("(") and expr.endswith(")") and expr.count("(") == expr.count(")"):
        expr = expr[1:-1].strip()

    # 展平 not(not(x))
    expr = re.sub(r"not\s*\(\s*not\s*\((.*?)\)\s*\)", r"\1", expr)

    return expr

def infer_guard_type(expr: str) -> str:
    """
    根据 guard 内容推断逻辑类型。
    """
    if not expr:
        return "unknown"
    if "dtype" in expr:
        return "dtype_check"
    if "device" in expr:
        return "device_check"
    if any(k in expr for k in ["shape", "size", "ndim", "dim"]):
        return "shape_check"
    if "None" in expr:
        return "existence_check"
    if re.search(r">|<|>=|<=|==|!=", expr):
        return "value_check"
    if re.search(r"in |not in ", expr):
        return "membership_check"
    if "not" in expr or "&&" in expr or "||" in expr:
        return "logical_check"
    return "boolean"

def normalize_guard(expr: str, src: str) -> dict:
    """
    将 guard 标准化为结构化形式。
    新增：检测反路径（not expr），拆分 lhs/rhs。
    """
    expr = clean_expr(expr)
    negated = False

    # 检测反路径
    if expr.startswith("not (") and expr.endswith(")"):
        negated = True
        expr = expr[4:-1].strip()

    # 拆分 lhs, op, rhs
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
        "negated": negated,
        "type": infer_guard_type(expr),
        "src": src
    }

def normalize_guards_stage(raw_guards: dict, api_name: str) -> dict:
    """
    规范化 Python + C++ guards，生成路径枚举友好格式。
    改进：增加反路径识别、统一清理、类型推断。
    """
    result = {"python": [], "cpp": []}

    for src in ["python", "cpp"]:
        guards = raw_guards.get(f"{src}_guards", [])
        if not guards:
            continue

        # 去重、过滤与参数无关的 guard
        guards = list({clean_expr(g) for g in guards if g.strip()})
        guards = filter_guards_by_args(guards, api_name)

        normalized = [normalize_guard(g, src) for g in guards]
        result[src].extend(normalized)

    # 合并为路径枚举格式
    for_path_enum = []
    for src, guards in result.items():
        for g in guards:
            for_path_enum.append({
                "expr": g["expr"],
                "src": src,
                "type": g["type"],
                "negated": g["negated"]
            })

    return {
        "normalized_guards": result,
        "for_path_enumeration": for_path_enum
    }

# =====================================================
# 批量提取并规范化 guards
# =====================================================

def generate_normalized_guards(api_names: list[str]):
    """
    批量提取并规范化 guards。
    基础版：
    - 支持 unknown 类型（不跳过）
    - 捕获最小异常防止中断
    - 立即写入文件避免进度丢失
    """
    output_path = f"../documentation/api_guards/{lib_name}_api_guards.json"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 若存在旧结果则加载
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    else:
        all_results = {}

    for api in api_names:
        torch_extract_api_source(api)

        if api in all_results:
            #print(f"⏩ Skipping already processed API: {api}")
            continue

        #print(f"\n[+] Processing API: {api}")

        try:
            # 分类并抽取 guards（包含 unknown 尝试）
            raw_guards = torch_extract_guards(api)

            # 规范化
            normalized = normalize_guards_stage(raw_guards, api)

            # 保存结果
            all_results[api] = {
                "normalized_guards": normalized["normalized_guards"],
                "for_path_enumeration": normalized["for_path_enumeration"]
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            print(f"    ✅ Saved normalized guards for {api}")

        except Exception as e:
            #print(f"    ❌ Error processing {api}: {e}")
            continue



# =====================================================
# 路径枚举
# =====================================================

# Python 层路径枚举核心
def enumerate_python_paths_core(api_name: str, api_data: dict):
    """
    改进版 Python 层路径枚举（融合 normalized_guards 的控制流 DFS）。

    修复版特点：
    - 每遇到 if / assert 时，分支执行后会继续执行后续语句（非立即 return）。
    - 每个路径完整覆盖从入口到 return / raise 的语句序列。
    - 保持与 normalized_guards 对齐。
    """
    normalized_guards = api_data.get("normalized_guards", {}).get("python", [])

    try:
        mod_name, attr_name = api_name.rsplit('.', 1)
        mod = __import__(mod_name, fromlist=[attr_name])
        py_obj = getattr(mod, attr_name)
        src = inspect.getsource(py_obj)
        tree = ast.parse(src)
    except Exception as e:
        #f"[WARN] enumerate_python_paths: cannot load source for {api_name}: {e}")
        return []

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break
    if func_node is None:
        return []

    def match_guard(expr: str, guards: list[dict]) -> str:
        for g in guards:
            if g.get("expr") == expr:
                return g["expr"]
        for g in guards:
            lhs = g.get("lhs", "")
            if lhs and lhs in expr:
                return g["expr"]
        return expr

    def unparse_cond(test_node: ast.AST) -> str:
        try:
            return ast.unparse(test_node)
        except Exception:
            return "<complex_expr>"

    paths = []

    def append_path(guards: list[str], path_type: str, ret_value: ast.AST | None):
        calls_cpp = False
        if path_type == "return_fun" and isinstance(ret_value, ast.Call):
            calls_cpp = True
        path_id = f"{api_name}_P{len(paths)+1}"
        paths.append({
            "id": path_id,
            "conjuncts": guards[:],
            "expr": " and ".join(guards) if guards else "",
            "src": ["python"],
            "path_type": path_type,
            "calls_cpp": calls_cpp,
            "complexity": len(guards),
            "sat": True
        })

    def exec_block(stmts: list[ast.stmt], guards_prefix: list[str]):
        guards = guards_prefix[:]
        i = 0
        n = len(stmts)
        while i < n:
            stmt = stmts[i]

            # If 分支（新增：继续执行剩余语句）
            if isinstance(stmt, ast.If):
                cond_raw = unparse_cond(stmt.test)
                cond_aligned = match_guard(cond_raw, normalized_guards)
                rest = stmts[i+1:]
                exec_block(list(stmt.body) + rest, guards + [cond_aligned])
                exec_block(list(stmt.orelse or []) + rest, guards + [f"not ({cond_aligned})"])
                return

            # Assert 分支（成功继续执行后续语句）
            if isinstance(stmt, ast.Assert):
                cond_raw = unparse_cond(stmt.test)
                cond_aligned = match_guard(cond_raw, normalized_guards)
                rest = stmts[i+1:]
                append_path(guards + [f"not ({cond_aligned})"], "raise", None)
                exec_block(rest, guards + [cond_aligned])
                return

            # Raise 终止路径
            if isinstance(stmt, ast.Raise):
                append_path(guards, "raise", None)
                return

            # Return 终止路径
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Call):
                    append_path(guards, "return_fun", stmt.value)
                else:
                    append_path(guards, "return", stmt.value)
                return

            i += 1

        append_path(guards, "return", None)

    exec_block(func_node.body, [])
    return paths

# python层路径枚举函数
def torch_enumerate_python_paths(json_path: str, api_name: str):
    """
    测试 enumerate_python_paths，输出更清晰的路径结构。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    if api_name not in all_results:
        #print(f"[ERROR] API '{api_name}' not found in {json_path}")
        return

    api_data = all_results[api_name]
    paths = enumerate_python_paths_core(api_name, api_data)
    return paths

# 利用 Joern + CPG 做 C++ 层路径枚举
def _joern_list_switches_with_order(joern: JoernShell, cpp_func_name: str):
    """
    从 Joern 中提取 C++ 函数内所有 switch(cond) 的条件与 case/default。
    返回:
      [
        {"order": <行号>, "cond": "x", "cases": ["1", "2", "default"]},
        {"order": <行号>, "cond": "y", "cases": ["10", "20", "default"]}
      ]
    """
    query_forhalf = f'cpg.method.name("{cpp_func_name}")'
    query_backhalf = r""".ast.isControlStructure.filter(_.code.startsWith("switch")).foreach { sw => 
        val ord = sw.lineNumber.getOrElse(-1)
        val cond = sw.code.split("\\(")(1).split("\\)")(0).trim
        val cases = sw.astChildren.flatMap(_.astChildren)
            .filter(n => n.code.startsWith("case") || n.code.startsWith("default"))
            .toSeq.map(n => if (n.code.startsWith("case"))
                n.code.split(":")(0).replace("case","").trim else "default")
        println(cond + "->" + cases.mkString(",") + "@" + ord)
    }"""
    
    raw = joern.send_command(query_forhalf + query_backhalf)

    # 匹配形如:  "x->1,2,default@32"
    pattern = re.compile(r"([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_, ]+)\s*@(\d+)")
    results = []
    for line in raw.splitlines():
        m = pattern.search(line)
        if m:
            cond = m.group(1).strip()
            cases = [c.strip() for c in m.group(2).split(",") if c.strip()]
            order = int(m.group(3))
            results.append({
                "order": order,
                "cond": cond,
                "cases": cases
            })

    results.sort(key=lambda x: x["order"])

    #print(f"[SWITCH DEBUG] {cpp_func_name}: 提取到 {len(results)} 个 switch 结构")
    #for sw in results:
        #print(f"  ↳ line {sw['order']}: switch({sw['cond']}) -> cases {sw['cases']}")
    return results

def _parse_control_structures(ctrl_raw: str):
    """从 Joern 输出中提取控制结构节点，标准化类型与条件。"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean = ansi_escape.sub('', ctrl_raw)
    clean = re.sub(r'joern>.*', '', clean)
    clean = re.sub(r'val res\d+:\s*List\[.*?\]\s*=\s*List\(', 'List(', clean)

    def _std_type(t: str) -> str:
        return re.sub(r'[^A-Z_]', '', t.upper())

    def _first_paren_chunk(s: str) -> str:
        """取第一个 () 内的内容；若失败，返回原串。"""
        m = re.search(r'\((.*)\)', s, flags=re.DOTALL)
        return (m.group(1).strip() if m else s.strip())

    def _first_arg_of_call(arglist: str) -> str:
        """获取调用形参串的首个参数（处理简单括号计数）。"""
        depth = 0
        buf = []
        for ch in arglist:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                break
            buf.append(ch)
        return ''.join(buf).strip()

    nodes = []
    current = {}

    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("ControlStructure("):
            current = {}
            continue

        if line.startswith("code ="):
            code = line.split("=", 1)[1].strip().rstrip(',').strip()
            current["code"] = code.strip('"')
            continue

        if "controlStructureType" in line:
            t = line.split("=", 1)[1].strip().rstrip(',').strip().strip('"')
            current["type"] = _std_type(t)
            continue

        if line.startswith("lineNumber"):
            try:
                current["order"] = int(re.search(r'\d+', line).group(0))
            except Exception:
                current["order"] = 10**9  # fallback
            continue

        if line.startswith("parserTypeName"):
            if not current:
                continue
            code = current.get("code", "")
            ctype = _std_type(current.get("type", "UNKNOWN"))
            order = current.get("order", 10**9)

            # 忽略无意义/顺序控制节点：ELSE/THROW/BREAK
            if ctype in {"ELSE", "THROW", "BREAK"}:
                current = {}
                continue

            # 解析条件
            cond = ""
            if "TORCH_CHECK" in code:
                inside = _first_paren_chunk(code)
                cond = _first_arg_of_call(inside)
                ctype = "IF_THROW"
            elif ctype == "FOR":
                header = _first_paren_chunk(code)  # init; cond; inc
                parts = [p.strip() for p in header.split(';')]
                cond = (parts[1] if len(parts) >= 2 else header) or header
            elif ctype in {"IF", "WHILE", "SWITCH"}:
                cond = _first_paren_chunk(code)
            else:
                # 其他未知结构，尽量保留 code 作为条件
                cond = code.strip()

            nodes.append({"order": order, "type": ctype, "cond": cond})
            current = {}

    nodes.sort(key=lambda x: x["order"])
    return nodes

# 除了 controlStructure，还需要显式提取 TORCH_CHECK 调用
def _append_torch_checks(joern, cpp_func_name, nodes):
    torch_check_query = (
        f'cpg.method.name("{cpp_func_name}").call.name("TORCH_CHECK").argument(1).code.l'
    )
    raw = joern.send_command(torch_check_query)
    conds = parse_scala_list(raw)
    order = 0
    for cond in conds:
        order += 1
        nodes.append({
            "order": 5 + order,  # 伪造一个较小 order 以保证排序在前
            "type": "IF_THROW",
            "cond": cond.strip()
        })
    return nodes

def build_cpp_paths(nodes, switches):
    """
    根据控制结构节点 + switch cases 构建所有执行路径。
    返回：每条路径是字符串列表，最后一项为 '→ return' 或 '→ raise'
    """
    # 把 switch 的真实 cases 建成索引：order -> cases
    case_map = {}
    for sw in switches or []:
        if sw.get("cases"):
            case_map[sw["order"]] = sw["cases"]

    active = [[]]     # 仍在继续扩展的路径（未结束）
    finished = []     # 已终止的路径（含 ->raise 或 ->return）

    for node in nodes:
        ntype, cond, order = node["type"], node["cond"], node["order"]

        next_active = []

        for path in active:
            # 若这条路径已经提前终止（理论上 active 里不该出现，但稳妥起见再拦一下）
            if path and path[-1] in ("→ raise", "→ return"):
                finished.append(path)
                continue

            if ntype == "IF_THROW":
                # True 分支：条件成立，继续执行
                next_active.append(path + [cond])
                # False 分支：条件不成立，立即异常终止
                finished.append(path + [f"not ({cond})", "→ raise"])

            elif ntype in {"IF", "FOR", "WHILE"}:
                # 两条分支都继续后续节点
                next_active.append(path + [cond])
                next_active.append(path + [f"not ({cond})"])

            elif ntype == "SWITCH":
                cases = case_map.get(order, ["default"])
                for c in cases:
                    next_active.append(path + [f"{cond} == {c}"])

            else:
                # 其他/顺序节点（很少见）：直接累加
                next_active.append(path + [cond])

        # 本轮结束后，更新 active
        active = next_active

    # 所有节点处理完毕：把仍未终止的 active 补上隐式 return
    for p in active:
        if not p or p[-1] not in ("→ raise", "→ return"):
            finished.append(p + ["→ return"])
        else:
            finished.append(p)

    return finished

def torch_enumerate_cpp_paths(api_name: str):
    """
    保持原接口 & 输出格式不变：打印每条路径并返回 paths(List[List[str]])。
    """
    # 你的测试固定入口
    cpp_func_name = torch_find_cpp_name(api_name)
    if not cpp_func_name:
        #print(f"[WARN] 无法找到 {api_name} 的 C++ 实现函数")
        return []

    # print(f"=== 提取 C++ 路径（Joern DFS）: {api_name} → {cpp_func_name} ===")

    joern = JoernShell(joern_bat_path)
    joern.send_command(f'open("{joern_project}")')

    # 控制结构
    ctrl_raw = joern.send_command(f'cpg.method.name("{cpp_func_name}").controlStructure.l')
    nodes = _parse_control_structures(ctrl_raw)
    nodes = _append_torch_checks(joern, cpp_func_name, nodes)
    nodes.sort(key=lambda x: x["order"])
    # 真实 switch cases
    switches = _joern_list_switches_with_order(joern, cpp_func_name)
    # print(switches)
    joern.send_command("exit")

    paths = build_cpp_paths(nodes, switches)

    return paths

# 合并两层路径枚举结果
def merge_python_cpp_paths(py_paths: list, cpp_paths: list, api_name: str):
    """
    将 Python 层与 C++ 层路径合并成完整执行路径空间。
    - Python 层中 path_type == "return_fun" 的路径会与所有 C++ 层路径组合；
    - 其他 Python 路径保持原状；
    - C++ 层路径结构为 [['x>=0', 'not(y>=0)', '→ return'], ...]。
    """
    merged = []
    path_id = 1
    if py_paths:
        for py_p in py_paths:
            ptype = py_p.get("path_type", "")
            py_src = py_p.get("src", ["py:unknown"])
            py_conds = py_p.get("conjuncts", [])
            if ptype == "return_fun":
                # 与 C++ 层路径做笛卡尔积
                if cpp_paths:
                    for cpp in cpp_paths:
                        cpp_conds = [c for c in cpp if not c.startswith("→")]
                        cpp_exit = "return" if any("→ return" in c for c in cpp) else "raise"
                        merged.append({
                            "id": f"{api_name}_{path_id}",
                            "conjuncts": py_conds + cpp_conds,
                            "src": py_src + ["cpp:testGuards"],
                            "path_type": cpp_exit,
                            "complexity": len(py_conds) + len(cpp_conds)
                        })
                        path_id += 1
                else:
                    merged.append({
                    "id": f"{api_name}_{path_id}",
                    "conjuncts": py_conds,
                    "src": py_src,
                    "path_type": ptype,
                    "complexity": len(py_conds)
                })
                path_id += 1
            else:
                # 直接保留 Python 路径
                merged.append({
                    "id": f"{api_name}_{path_id}",
                    "conjuncts": py_conds,
                    "src": py_src,
                    "path_type": ptype,
                    "complexity": len(py_conds)
                })
                path_id += 1
    else:
        # 仅 C++ 层路径
        if cpp_paths is None:
            return merged
        for cpp in cpp_paths:
            cpp_conds = [c for c in cpp if not c.startswith("→")]
            cpp_exit = "return" if any("→ return" in c for c in cpp) else "raise"
            merged.append({
                "id": f"{api_name}_{path_id}",
                "conjuncts": cpp_conds,
                "src": ["cpp:testGuards"],
                "path_type": cpp_exit,
                "complexity": len(cpp_conds)
            })
            path_id += 1
    if not merged:
        merged.append({
            "id": f"{api_name}_{path_id}",
            "conjuncts": [],
            "src": [],
            "path_type": "return",
            "complexity": 0
        })

    return merged

# =====================================================
# 获取源码
# =====================================================
def torch_extract_api_source(api_name: str):
    """
    提取给定 PyTorch API 的 Python 源码和对应 C++ 源码。
    统一保存到一个 JSON 文件，key 为 api_name。
    """
    output_path = f"../documentation/api_src_code/{lib_name}_api_sources.json"
    pytorch_root = "C:/Users/86184/Desktop/Papers/dl_lib/pytorch-2.5.1"

    # ========== 1️⃣ Python 源码提取 ==========
    py_file = None
    py_start, py_end, py_code = None, None, ""

    try:
        target = eval(api_name)  # 反射函数对象
        src_file = inspect.getsourcefile(target)
        src_lines, start_line = inspect.getsourcelines(target)
        py_file = os.path.relpath(src_file, pytorch_root)

        # 去掉 docstring
        src_code = "".join(src_lines)
        # 去掉函数头和缩进
        body = textwrap.dedent(src_code)
        # 删除首个三引号字符串 """...""" 或 '''...'''
        body = re.sub(r'^[ \t]*[ruRU]*[\'"]{3}[\s\S]*?[\'"]{3}\n?', '', body, count=1, flags=re.MULTILINE)
        # 去掉前导空行
        body = body.lstrip("\n")

        py_start = start_line
        py_end = start_line + len(src_lines) - 1
        py_code = body

        # print(f"[PYTHON] {api_name} -> {py_file}:{py_start}-{py_end}")
    except Exception as e:
        print(f"[WARN] 无法提取 Python 源码: {api_name}, error={e}")

    # ========== 2️⃣ C++ 源码提取 ==========
    cpp_func_name = torch_find_cpp_name(api_name)
    cpp_file, cpp_start, cpp_end, cpp_code = None, None, None, ""

    joern = JoernShell(joern_bat_path)
    joern.send_command(f'open("{joern_project}")')

    #print(f"[CPP] 提取 {cpp_func_name} 的源码")

    query_meta = f'''
        cpg.method.name("{cpp_func_name}").foreach {{
        m =>
            val fn  = m.filename
            val ln1 = m.lineNumber.getOrElse(-1).toString
            val ln2 = m.lineNumberEnd.getOrElse(-1).toString
            println("META_BEGIN" + fn + "||" + ln1 + "||" + ln2 + "META_END")
        }}
        '''
    meta_raw = joern.send_command(query_meta)
    m = re.search(r'META_BEGIN(.*?)META_END', meta_raw, re.DOTALL)
    if m:
        file_rel, start_line_s, end_line_s = m.group(1).split("||")
        cpp_file = file_rel.strip().replace("\\", "/")
        cpp_start, cpp_end = int(start_line_s), int(end_line_s)
        abs_cpp = (Path(pytorch_root) / cpp_file).resolve()
        if abs_cpp.exists():
            with open(abs_cpp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                cpp_code = "".join(lines[cpp_start-1:cpp_end])
        else:
            #print(f"[WARN] 找不到 {abs_cpp}，回退为 Joern 输出")
            code_raw = joern.send_command(f'cpg.method.name("{cpp_func_name}").code.l')
            ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            cpp_code = "\n".join([
                l.strip() for l in ansi.sub('', code_raw).splitlines()
                if l.strip() and not l.startswith("joern>")
            ])

    joern.send_command("exit")

    # ========== 3️⃣ 汇总并保存 ==========
    api_data = {
        "python": {
            "file": py_file,
            "start_line": py_start,
            "end_line": py_end,
            "code": py_code
        },
        "cpp": {
            "function": cpp_func_name,
            "file": cpp_file,
            "start_line": cpp_start,
            "end_line": cpp_end,
            "code": cpp_code
        }
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}
    else:
        data = {}

    data[api_name] = api_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # print(f"[DONE] {api_name} 源码已保存至 {output_path}")
    return api_data


# 在当前文件夹生成所有 API 的 guards 和路径枚举结果


if __name__ == "__main__":
    api_def_path = f"../documentation/lib_api/{lib_name}_APIdef.txt"

    # arg_space 的生成依赖于 api_guards 
    # api_guards 的生成依赖于本文件中的 guard 提取函数
    j = 0
    save_path = f"../documentation/arg_space/{lib_name}_arg_space_{j}.json"

    api_names = read_file(api_def_path)
    # 生成api_guards.json 和 api_sources.json
    generate_normalized_guards(api_names)


    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            grouped_results = json.load(f)
    else:
        grouped_results = {}

    for api_name in api_names:
        if api_name in grouped_results:
            print(f"[⏭️ Skip] {api_name} 已存在于结果中，跳过。")
            continue

        ppaths = torch_enumerate_python_paths(f"../documentation/api_guards/{lib_name}_api_guards.json", api_name)
        cpaths = torch_enumerate_cpp_paths(api_name)
        merged_paths = merge_python_cpp_paths(ppaths, cpaths, api_name)

        grouped_results[api_name] = grouped_results.get(api_name, [])
        grouped_results[api_name].extend(merged_paths)
        if is_file_too_large(save_path, max_size_mb=512):
            j+=1
            save_path = f"../documentation/arg_space/{lib_name}_arg_space_{j}.json"
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(grouped_results, f, indent=4, ensure_ascii=False)
            print(f"[💾 Saved] {api_name}: {len(merged_paths)} 条路径已写入。")
        except Exception as e:
            print(f"[❌ Save Error] 写入文件失败 ({api_name}): {e}")

    print("\n✅ 所有 API 已处理完毕，结果保存在：", save_path)







