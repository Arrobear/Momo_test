# Stage 2 Function 通用化改造说明

## 改造目标
将原本专门为 PyTorch 设计的 guard 提取工具改造成支持任意第三方库的通用工具。

## 主要修改

### 1. 配置部分（第 10-11 行）
**修改前：**
```python
TORCH_PATH = Path("C:/Users/86184/Desktop/Papers/dl_lib/pytorch-2.5.1")
YAML_PATH = TORCH_PATH / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
```

**修改后：**
```python
# 通用库配置 - 根据 config.py 中的 lib_name 动态设置
# PyTorch 特定配置（仅当 lib_name == "torch" 时使用）
TORCH_PATH = Path("C:/Users/86184/Desktop/Papers/dl_lib/pytorch-2.5.1") if lib_name == "torch" else None
YAML_PATH = TORCH_PATH / "aten" / "src" / "ATen" / "native" / "native_functions.yaml" if TORCH_PATH else None
```

**说明：** 只有当处理 PyTorch 库时才设置这些路径，避免其他库运行时出错。

---

### 2. API 分类函数

#### 新增通用分类函数 `generic_api_classify()`
- 使用 Python 反射机制判断 API 类型
- 支持：class, function, builtin, method, callable, unknown
- 不依赖任何特定库的配置文件

#### 改造 `torch_api_classify()`
- 保留为向后兼容的别名
- 对于非 torch 库，直接调用 `generic_api_classify()`
- 对于 torch 库，在通用分类基础上增加 PyTorch 特定逻辑（如 OpOverload、factory 函数等）

---

### 3. C++ 函数名查找

#### 修改 `torch_find_cpp_name()`
- 增加检查：只有当 API 是 torch 且 YAML 文件存在时才执行
- 对于非 torch 库，直接返回 `None`
- 避免因缺少 YAML 文件而导致的 FileNotFoundError

---

### 4. Python Guards 提取

#### 新增 `generic_extract_python_guards()`
- 通用的 Python 层 guard 提取逻辑
- 支持 if/assert/函数调用等常见 guard 模式
- 扩展了断言函数识别：`_assert`, `_check`, `check`, `validate`

#### 改造 `torch_extract_python_guards()`
- 改为调用 `generic_extract_python_guards()` 的别名

---

### 5. 各类型 API 的 Guards 提取

所有提取函数都进行了通用化改造：

#### `generic_extract_function_guards()` / `torch_extract_function_guards()`
- Python 层：使用通用提取
- C++ 层：仅对 torch 库尝试提取

#### `generic_extract_builtin_guards()` / `torch_extract_builtin_guards()`
- Python 层：使用通用提取
- C++ 层：仅对 torch 库尝试提取

#### `generic_extract_factory_guards()` / `torch_extract_factory_guards()`
- Python 层：使用通用提取
- C++ 层：仅对 torch 库尝试提取

#### `generic_extract_class_guards()` / `torch_extract_class_guards()`
- 递归分析 class 的 forward/__call__/__init__ 方法
- 收集 Python 层的 if guards
- 对于 torch 库，尝试提取调用的 C++ 函数的 guards

#### `generic_extract_unknown_guards()` / `torch_extract_unknown_guards()`
- 尽力而为地提取 Python 层 guards
- 对于 torch 库，尝试通过 YAML 映射到 C++ 实现

---

### 6. 源码提取

#### 新增 `generic_extract_api_source()`
- 自动检测库的根目录
- 提取 Python 源码（所有库通用）
- 提取 C++ 源码（仅 PyTorch）
- 使用相对路径保存文件位置

#### 改造 `torch_extract_api_source()`
- 改为调用 `generic_extract_api_source()` 的别名

---

## 使用方法

### 1. 配置库名称
在 `config.py` 中设置：
```python
lib_name = "mimesis"  # 或其他库名
```

### 2. 准备 API 列表
创建文件：`../documentation/lib_api/{lib_name}_APIdef.txt`
每行一个 API 名称，例如：
```
mimesis.Person.full_name
mimesis.Address.city
```

### 3. 运行提取
```bash
conda run --no-capture-output -n momo_test python -u stage_2_function.py
```

### 4. 输出文件
- `../documentation/api_guards/{lib_name}_api_guards.json` - API guards
- `../documentation/api_src_code/{lib_name}_api_sources.json` - API 源码
- `../documentation/arg_space/{lib_name}_arg_space_0.json` - 参数空间

---

## 兼容性说明

### 向后兼容
- 所有 `torch_*` 函数都保留为别名，调用对应的 `generic_*` 函数
- 对于 PyTorch 库，行为与之前完全一致
- 现有代码无需修改即可继续使用

### 新库支持
- 对于非 PyTorch 库：
  - 只提取 Python 层 guards
  - 不尝试提取 C++ 层（因为大多数库没有 C++ 实现或没有 Joern 项目）
  - 自动跳过所有 PyTorch 特定的逻辑

---

## 注意事项

1. **C++ 层提取**：只有 PyTorch 支持 C++ 层 guard 提取，其他库会自动跳过
2. **Joern 项目**：如果要为其他库提取 C++ guards，需要：
   - 准备该库的 C++ 源码
   - 使用 Joern 创建 CPG 项目
   - 在 `config.py` 中配置 `joern_project` 和 `joern_bat_path`
3. **API 分类**：不同库的 API 结构可能不同，分类结果仅供参考

---

## 测试建议

1. 先用小规模 API 列表测试（5-10 个 API）
2. 检查生成的 JSON 文件格式是否正确
3. 验证 Python guards 是否被正确提取
4. 对于有 C++ 实现的库，验证 C++ guards 提取逻辑

---

## 故障排查

### 问题：FileNotFoundError: native_functions.yaml
**原因：** 代码尝试访问 PyTorch 的 YAML 文件，但文件不存在或路径错误
**解决：** 
- 确保 `config.py` 中 `lib_name` 设置正确
- 如果不是 torch 库，此错误已被修复，不应再出现

### 问题：无法提取 Python 源码
**原因：** API 可能是 C 扩展或内建函数
**解决：** 这是正常情况，代码会自动跳过并继续处理

### 问题：guards 为空
**原因：** API 实现中可能没有条件判断语句
**解决：** 这是正常情况，某些简单 API 确实没有 guards

---

## 后续优化建议

1. 支持更多库的 C++ 层提取（需要准备 Joern 项目）
2. 增强 API 分类逻辑，支持更多特殊类型
3. 优化 guard 规范化逻辑，提高准确性
4. 添加更详细的日志输出，便于调试
