import torch
import numpy as np

# ==========================================
# 1. 基础类型生成器 (Base Generators)
# ==========================================

def _gen_tensor(param_info):
    """生成 Tensor 构造代码的字符串列表，支持存入 JSON"""
    candidates = []
    shape_min = param_info.get("shape_min", [1])
    shape_max = param_info.get("shape_max", [4])
    dtypes_str = param_info.get("dtypes", ["torch.float32"])
    
    # 形状提取逻辑保持不变
    s_min = tuple(shape_min[-1] if shape_min and isinstance(shape_min[0], list) else shape_min)
    s_max = tuple(shape_max[-1] if shape_max and isinstance(shape_max[0], list) else shape_max)
    s_min = tuple([d if d is not None else 2 for d in s_min])
    s_max = tuple([d if d is not None else 2 for d in s_max])
    
    shapes = list(set([s_min, s_max]))
    
    for dt_str in dtypes_str:
        if not dt_str.startswith("torch."): continue
        try:
            # 这里需要 eval dt_str 来判断类型属性 (floating, complex等)
            dt = eval(dt_str) 
            
            for shape in shapes:
                if dt.is_floating_point:
                    # 使用 f-string 构造代码字符串
                    cmd = f"torch.randn({list(shape)}, dtype={dt_str})"
                elif dt.is_complex:
                    cmd = f"torch.randn({list(shape)}, dtype=torch.float32).to({dt_str})"
                elif dt == torch.bool:
                    cmd = f"torch.randint(0, 2, {list(shape)}, dtype={dt_str})"
                else:
                    cmd = f"torch.randint(1, 10, {list(shape)}, dtype={dt_str})"
                
                candidates.append(cmd)
        except Exception:
            continue
            
    return candidates if candidates else ["torch.tensor([1.0])"]

def _gen_ndarray(param_info):
    """生成 numpy.ndarray"""
    candidates = []
    dtypes_str = param_info.get("dtypes", ["numpy.float32"])
    for dt_str in dtypes_str:
        if not dt_str.startswith("numpy."): continue
        try:
            dt = eval(dt_str)
            candidates.append(np.ones((2, 2), dtype=dt))
        except Exception:
            continue
    return candidates if candidates else [np.array([1.0])]

def _gen_int(param_info):
    """生成整数，提取 min, max"""
    min_val = param_info.get("min", 0)
    max_val = param_info.get("max", 10)
    if min_val is None or min_val == "null": min_val = 0
    if max_val is None or max_val == "null": max_val = 10
    
    try:
        min_v, max_v = int(float(min_val)), int(float(max_val))
        if min_v <= max_v:
            mid_v = (min_v + max_v) // 2
            return list(set([min_v, mid_v, max_v]))
        return [0]
    except:
        return [0, 1]

def _gen_float(param_info):
    """生成浮点数"""
    min_val = param_info.get("min", 0.0)
    max_val = param_info.get("max", 1.0)
    if min_val is None or min_val == "null": min_val = 0.0
    if max_val is None or max_val == "null": max_val = 1.0
    
    try:
        min_v, max_v = float(min_val), float(max_val)
        if min_v <= max_v:
            mid_v = (min_v + max_v) / 2.0
            return list(set([min_v, mid_v, max_v]))
        return [0.0]
    except:
        return [0.0, 1.0]

def _gen_bool(param_info):
    return [True, False]

def _gen_str(param_info):
    """生成字符串，优先使用 choices 和 default"""
    if "choices" in param_info and param_info["choices"]:
        return [c for c in param_info["choices"] if isinstance(c, str) and c != "null"]
    if "default" in param_info and param_info["default"] != "null":
        return [param_info["default"]]
    return ["", "test_string"]

def _gen_tuple(param_info, element_type="int"):
    """生成元组"""
    val = _gen_int(param_info)[0] if element_type == "int" else 1.0
    return [(val,), (val, val)]

def _gen_list(param_info, element_type="int"):
    """生成列表"""
    val = _gen_int(param_info)[0] if element_type == "int" else 1.0
    return [[val], [val, val]]

def _gen_choices(param_info):
    """针对枚举类型的直接映射"""
    candidates = []
    for choice in param_info.get("choices", []):
        if choice == "null":
            candidates.append(None)
        elif isinstance(choice, str) and (choice.startswith("torch.") or choice.startswith("numpy.")):
            try: candidates.append(eval(choice))
            except: candidates.append(choice)
        else:
            candidates.append(choice)
    return candidates

def _gen_mock_object(obj_type):
    """生成模拟对象 (如 module, callable, dict, file-like object)"""
    if "dict" in obj_type.lower() or "mapping" in obj_type.lower():
        return [{"mock_key": "mock_val"}]
    if "callable" in obj_type.lower() or "function" in obj_type.lower():
        return [lambda x: x]
    if "file" in obj_type.lower() or "path" in obj_type.lower():
        return ["./mock_path.txt"]
    if "size" in obj_type.lower():
        return [torch.Size([2, 2])]
    if "device" in obj_type.lower():
        return [torch.device("cpu")]
    return ["mock_object"]


# ==========================================
# 2. 核心入口路由 (Main Dispatcher)
# ==========================================

def generate_sample_param(api_name, param_name, param_info):
    """
    根据 API 的边界规范，显式路由并生成满足约束的测试输入。
    通过多重 if-elif 覆盖所有情况，便于独立维护。
    """
    p_type = str(param_info.get("type", "")).strip()
    candidates = []

    # ---------------------------------------------------------
    # 分支 1：纯张量与数组类 (Tensor / ndarray)
    # ---------------------------------------------------------
    if p_type in ["Tensor", "LongTensor", "IntTensor", "Tensors", "array_like"]:
        candidates.extend(_gen_tensor(param_info))
    
    elif p_type in ["Optional[Tensor]", "Tensor, optional", "Tensor or null", "Optional[LongTensor]"]:
        candidates.extend(_gen_tensor(param_info))
        candidates.append(None)
        
    elif p_type == "numpy.ndarray":
        candidates.extend(_gen_ndarray(param_info))
        
    elif p_type in ["list[Tensor]", "sequence of Tensors", "Iterable[Tensor]", "Iterable[Tensor] or Tensor", "Tensor or list of tensors"]:
        tensors = _gen_tensor(param_info)
        candidates.extend(tensors)           # 单个 Tensor
        candidates.append([tensors[0]])      # Tensor 列表
        
    elif p_type == "tuple of two tensors":
        tensors = _gen_tensor(param_info)
        candidates.append((tensors[0], tensors[0]))
        
    elif p_type == "Optional[Tuple[Tensor, Tensor]]":
        tensors = _gen_tensor(param_info)
        candidates.append((tensors[0], tensors[0]))
        candidates.append(None)

    # ---------------------------------------------------------
    # 分支 2：纯数值与基础类型 (Int / Float / Bool / Str)
    # ---------------------------------------------------------
    elif p_type in ["int", "int..."]:
        candidates.extend(_gen_int(param_info))
        
    elif p_type == "Optional[int]":
        candidates.extend(_gen_int(param_info))
        candidates.append(None)
        
    elif p_type in ["float", "Number", "Scalar"]:
        candidates.extend(_gen_float(param_info))
        
    elif p_type == "Optional[float]":
        candidates.extend(_gen_float(param_info))
        candidates.append(None)
        
    elif p_type == "bool":
        candidates.extend(_gen_bool(param_info))
        
    elif p_type in ["Optional[bool]", "bool or null", "bool|null"]:
        candidates.extend(_gen_bool(param_info))
        candidates.append(None)
        
    elif p_type == "str":
        candidates.extend(_gen_str(param_info))
        
    elif p_type == "Optional[str]":
        candidates.extend(_gen_str(param_info))
        candidates.append(None)

    # ---------------------------------------------------------
    # 分支 3：元组与列表类 (Tuple / List)
    # ---------------------------------------------------------
    elif p_type in ["tuple", "tuple[int]", "tuple of int", "list[int] or tuple[int]", "list or tuple"]:
        candidates.extend(_gen_tuple(param_info, element_type="int"))
        
    elif p_type in ["tuple or null", "Optional[tuple]"]:
        candidates.extend(_gen_tuple(param_info))
        candidates.append(None)
        
    elif p_type == "list[int]":
        candidates.extend(_gen_list(param_info, element_type="int"))

    # ---------------------------------------------------------
    # 分支 4：二元/三元联合类型 (Union / OR)
    # ---------------------------------------------------------
    elif p_type in ["Union[int, Tuple[int], str]", "int, tuple, or str", "Union[int, Tuple[int, int], str]", "Union[int, Tuple[int, int, int], str]"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_tuple(param_info, element_type="int"))
        candidates.extend(_gen_str(param_info))
        
    elif p_type in ["Union[int, Tuple[int, int]]", "Union[int, Tuple[int, int, int]]", "Union[int, Tuple[int]]", "int or tuple", "tuple or int", "int or tuple of ints"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_tuple(param_info, element_type="int"))
        
    elif p_type in ["Optional[Union[int, Tuple[int]]]", "Optional[Union[int, Tuple[int, int]]]", "Union[int, Tuple[int, int], null]", "int or tuple of ints or null"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_tuple(param_info, element_type="int"))
        candidates.append(None)
        
    elif p_type in ["Union[float, Tuple[float, float]]", "Union[float, Tuple[float], Tuple[float, float], Tuple[float, float, float]]"]:
        candidates.extend(_gen_float(param_info))
        candidates.extend(_gen_tuple(param_info, element_type="float"))
        
    elif p_type in ["Optional[Union[float, Tuple[float, float]]]", "Optional[Union[float, Tuple[float], Tuple[float, float], Tuple[float, float, float]]]"]:
        candidates.extend(_gen_float(param_info))
        candidates.extend(_gen_tuple(param_info, element_type="float"))
        candidates.append(None)
        
    elif p_type in ["float or Tensor", "Tensor or Number", "Number or Tensor", "Tensor or Scalar", "Tensor or float"]:
        candidates.extend(_gen_float(param_info))
        candidates.extend(_gen_tensor(param_info))
        
    elif p_type in ["int or Tensor", "Tensor or list(int)"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_tensor(param_info))
        
    elif p_type in ["Union[int, Tensor, List[int], Tuple[int]]", "int or tuple or list or Tensor", "int or Tuple[List[int], List[int]] or Tensor", "int or Tuple[List[int], List[int]] or List[List[int]] or Tensor", "int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_tensor(param_info))
        candidates.extend(_gen_list(param_info, element_type="int"))
        candidates.extend(_gen_tuple(param_info, element_type="int"))

    elif p_type in ["Union[int, List[int]]", "int or Tuple[List[int], List[int]] or List[List[int]]"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_list(param_info, element_type="int"))
        
    elif p_type == "Union[int, str]":
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_str(param_info))

    # ---------------------------------------------------------
    # 分支 5：PyTorch 特定对象与配置属性
    # ---------------------------------------------------------
    elif p_type in ["torch.dtype", "dtype", "type or string", "Optional[torch.dtype]", "Optional[dtype]"]:
        candidates.extend(_gen_choices(param_info))
        if "Optional" in p_type: candidates.append(None)
        
    elif p_type in ["torch.device", "device", "Optional[torch.device]", "Optional[device]", "Optional[Union[torch.device, str]]", "optional[Union[str, torch.device]]"]:
        candidates.extend(_gen_choices(param_info))
        if "Optional" in p_type or "optional" in p_type: candidates.append(None)
        
    elif p_type in ["torch.layout", "layout", "Optional[torch.layout]"]:
        candidates.extend(_gen_choices(param_info))
        if "Optional" in p_type: candidates.append(None)
        
    elif p_type in ["torch.memory_format", "Optional[torch.memory_format]"]:
        candidates.extend(_gen_choices(param_info))
        if "Optional" in p_type: candidates.append(None)
        
    elif p_type in ["torch.Size", "Variable number of torch.Size objects", "Union[torch.Size, Tuple, List, NamedShape]"]:
        candidates.extend(_gen_mock_object("size"))
        
    elif p_type == "int or list or torch.Size":
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_list(param_info, element_type="int"))
        candidates.extend(_gen_mock_object("size"))
        
    elif p_type == "Optional[torch.Generator]":
        candidates.extend(_gen_choices(param_info))
        
    elif p_type in ["List[Union[int, torch.device]]", "Optional[Union[int, torch.device]]", "Optional[List[Union[int, torch.device]]]"]:
        candidates.extend(_gen_int(param_info))
        candidates.extend(_gen_mock_object("device"))
        if "Optional" in p_type: candidates.append(None)

    # ---------------------------------------------------------
    # 分支 6：复杂结构与模块 (Modules / Callables / Dicts)
    # ---------------------------------------------------------
    elif p_type in ["callable", "optional[Callable]", "Union[str, callable]", "Callable[[Module, Any], Optional[Any]]"]:
        candidates.extend(_gen_mock_object("callable"))
        if "str" in p_type: candidates.extend(_gen_str(param_info))
        if "optional" in p_type.lower(): candidates.append(None)
        
    elif p_type in ["dict", "Optional[dict]", "Mapping[str, Module]"]:
        candidates.extend(_gen_mock_object("dict"))
        if "Optional" in p_type: candidates.append(None)
        
    elif p_type in ["Union[file-like object, str, os.PathLike]", "str|PathLike|file-like object", "file-like object or string/PathLike", "str or PathLike"]:
        candidates.extend(_gen_mock_object("file"))
        
    elif p_type in ["nn.Module", "optional[nn.Module]", "module", "TransformerEncoderLayer", "optional[LayerNorm]", "TransformerEncoderLayer instance", "Optional[LayerNorm instance]", "LayerNorm", "TransformerDecoderLayer"]:
        candidates.extend(_gen_mock_object("module"))
        if "optional" in p_type.lower(): candidates.append(None)

    elif p_type == "callable|torch.device|string|dict|null":
        candidates.extend(_gen_mock_object("callable"))
        candidates.extend(_gen_mock_object("device"))
        candidates.extend(_gen_mock_object("dict"))
        candidates.extend(_gen_str(param_info))
        candidates.append(None)
        
    elif p_type == "PackedSequence":
        # 如果需要精确测试可以构造实际的 PackedSequence，这里给个占位
        candidates.extend(_gen_tensor(param_info))

    # ---------------------------------------------------------
    # 分支 7：通用与兜底 (Optional / Any / Object)
    # ---------------------------------------------------------
    elif p_type in ["optional", "any", "Object", "Sequence", "iterable", "complex"]:
        if "choices" in param_info:
            candidates.extend(_gen_choices(param_info))
        else:
            candidates.extend(_gen_mock_object("object"))
        if p_type == "optional": candidates.append(None)
        if param_info.get("optional") is True: candidates.append(None)
        return "complex"  # 标记为复杂类型，后续使用模型生成更多样本

    else:
        # 万一出现了未捕获的新类型，使用兜底逻辑
        print(f"Warning: Unhandled param type: {p_type} for {param_name}")
        if "choices" in param_info:
            candidates.extend(_gen_choices(param_info))
        candidates.append(None)

    # ---------------------------------------------------------
    # 3. 安全去重 (Deduplication)
    # ---------------------------------------------------------
    final_candidates = []
    for c in candidates:
        is_dup = False
        for fc in final_candidates:
            try:
                if type(c) == type(fc):
                    if isinstance(c, torch.Tensor):
                        if c.shape == fc.shape and c.dtype == fc.dtype and torch.equal(c, fc):
                            is_dup = True
                            break
                    elif isinstance(c, np.ndarray):
                        if c.shape == fc.shape and c.dtype == fc.dtype and np.array_equal(c, fc):
                            is_dup = True
                            break
                    elif c == fc:
                        is_dup = True
                        break
            except Exception:
                pass
        if not is_dup:
            final_candidates.append(c)

    return final_candidates

# 测试桩
if __name__ == "__main__":
    test_param = {
        "type": "Union[int, Tuple[int, int], str]",
        "choices": ["valid", "same"],
        "min": 0,
        "max": 10
    }
    test_param2 =     {
        "type": "Tensor",
        "shape_min": [
            1,
            1,
            1
        ],
        "shape_max": [
            8,
            16,
            1024
        ],
        "dtypes": [
            "torch.float32",
            "torch.float64",
            "torch.complex64"
        ]
    }
    print(generate_sample_param("test_api", "test_param", test_param2))