import importlib
import inspect
import sys
import types
import warnings
import glom
def extract_library_apis(package_name, ignore_private=True):
    """
    动态提取指定Python库中所有可调用的API及其定义（Signature）。
    
    参数:
        package_name (str): 第三方库的名称，如 'torch', 'tensorflow', 'numpy'
        ignore_private (bool): 是否忽略以 '_' 开头的私有API
        
    返回:
        list of tuples: [(api_name, api_definition), ...]
    """
    try:
        # 忽略加载库时可能产生的第三方警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root_module = importlib.import_module(package_name)
    except ImportError as e:
        print(f"[-] 无法导入库 '{package_name}': {e}")
        return []

    visited_objects = set()
    api_collection = {}

    def walk_namespace(obj, current_path):
        """递归遍历命名空间"""
        # 防止循环引用导致无限递归
        obj_id = id(obj)
        if obj_id in visited_objects:
            return
        visited_objects.add(obj_id)

        # 遍历当前对象的所有属性
        for attr_name in dir(obj):
            # 过滤私有方法和属性
            if ignore_private and attr_name.startswith('_') and not attr_name.startswith('__'):
                continue

            try:
                attr_val = getattr(obj, attr_name)
            except Exception:
                # 某些属性在获取时可能会抛出异常（例如某些动态计算的property）
                continue

            full_path = f"{current_path}.{attr_name}" if current_path else attr_name

            # 1. 如果是模块，检查是否属于目标库，然后递归
            if inspect.ismodule(attr_val):
                mod_name = getattr(attr_val, '__name__', '')
                # 必须限制在目标包内，防止漫游到系统内置库（如 os, sys）
                if mod_name.startswith(package_name):
                    walk_namespace(attr_val, full_path)
            
            # 2. 如果是可调用对象（函数、类、方法）
            elif callable(attr_val):
                
                # (可选) 过滤异常类，防止提取出 Error
                if isinstance(attr_val, type) and issubclass(attr_val, BaseException):
                    continue

                # 尝试过滤掉从其他基础库导入的 callable
                mod_name = getattr(attr_val, '__module__', None)
                if mod_name and not mod_name.startswith(package_name) and mod_name != 'builtins':
                    continue

                try:
                    # 尝试获取精确的 API 定义
                    sig = inspect.signature(attr_val)
                    api_def = f"{full_path}{sig}"
                except (ValueError, TypeError):
                    # 【核心修改】：如果遇到无法解析签名的情况 (C-Extension/内置函数/特殊对象)
                    # 不再保留，直接跳过当前这个 API
                    continue

                # 存入字典去重
                if full_path not in api_collection:
                    api_collection[full_path] = api_def

    print(f"[*] 正在分析库: {package_name} ...")
    walk_namespace(root_module, package_name)
    
    # 转换为列表并按字母顺序排序
    sorted_apis = sorted(list(api_collection.items()), key=lambda x: x[0])
    return sorted_apis

if __name__ == "__main__":
    # 测试用例：提取 json 库的 API（替换为您需要测试的第三方库如 'torch', 'tensorflow'）
    # 注意：提取像 tf 这种巨型库可能需要几秒到十几秒的时间，并且会消耗一定内存
    target_lib = 'requests' 
    
    apis = extract_library_apis(target_lib)
    
    print(f"[*] 共提取到 {len(apis)} 个可调用 API。\n")
    
    # 将结果输出到文件，方便后续测试工作使用
    output_file = f"../documentation/lib_api/{target_lib}_APIdef.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for api_name, api_def in apis:
            # f.write(f"API Name: {api_name}\n")
            f.write(f"{api_def}\n")
            # f.write("-" * 50 + "\n")
            
    print(f"[*] 提取完成！结果已保存至 {output_file}")