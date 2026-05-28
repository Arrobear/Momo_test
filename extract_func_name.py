import importlib
import inspect
import warnings
from config import lib_name

def extract_logic_methods(package_name):
    """
    专为逻辑测试定制：
    仅提取模块级的独立函数，以及类中封装的具体方法。
    不提取类本身的实例化签名（如 ClassName(*args)）。
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root_module = importlib.import_module(package_name)
    except ImportError as e:
        print(f"[-] 无法导入库: {e}")
        return []

    apis = {}
    visited = set()

    # 必须要屏蔽的结构性指针和防止重复提取的初始化方法
    STRUCTURAL_DUNDERS = {
        '__class__', '__base__', '__bases__', '__mro__', '__subclasses__',
        '__dict__', '__doc__', '__annotations__', '__name__', '__qualname__',
        '__globals__', '__closure__', '__code__', '__dir__', '__weakref__',
        '__init__', '__new__'
    }

    def is_in_target_package(obj):
        """严格校验对象是否为当前包的原生对象"""
        mod = getattr(obj, '__module__', None)
        if not mod and hasattr(obj, '__objclass__'):
            mod = getattr(obj.__objclass__, '__module__', None)
        return bool(mod and mod.startswith(package_name))

    def walk(obj, current_path):
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        try:
            members = inspect.getmembers(obj)
        except Exception:
            return

        for name, member in members:
            # 1. 拦截单下划线的纯私有方法，放行双下划线的魔术方法（支持测试库作者重写的底层行为）
            if name.startswith('_') and not name.startswith('__'):
                continue

            # 2. 拦截非业务类的结构性魔术方法
            if name in STRUCTURAL_DUNDERS:
                continue

            path = f"{current_path}.{name}" if current_path else name

            # 节点类型 1：子模块 -> 继续深入
            if inspect.ismodule(member):
                if getattr(member, '__name__', '').startswith(package_name):
                    walk(member, path)
            
            # 节点类型 2：类 -> 【核心修改】只递归，不提取签名
            elif inspect.isclass(member):
                if not is_in_target_package(member) or issubclass(member, BaseException):
                    continue
                # 不再执行 inspect.signature(member)，直接深入内部找方法
                walk(member, path)

            # 节点类型 3：函数/方法 -> 测试用例的核心目标，提取签名
            elif callable(member):
                if not is_in_target_package(member):
                    continue
                
                try:
                    sig = inspect.signature(member)
                    apis[path] = f"{path}{sig}"
                except (ValueError, TypeError):
                    pass

    print(f"[*] 正在深度扫描库: {package_name} ...")
    walk(root_module, package_name)
    
    return sorted(list(apis.items()), key=lambda x: x[0])

if __name__ == "__main__":
    target_lib = lib_name 
    
    apis = extract_logic_methods(target_lib)
    
    print(f"[*] 共提取到 {len(apis)} 个用于逻辑测试的方法/函数。\n")
    
    import os
    os.makedirs("../documentation/lib_api", exist_ok=True)
    output_file = f"../documentation/lib_api/{target_lib}_APIdef.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for api_name, api_def in apis:
                f.write(f"{api_def}\n")
        print(f"[*] 提取完成！结果已保存至 {output_file}")
    except IOError as e:
        print(f"[-] 文件保存失败: {e}")