from config import *
from stage_1_function import *
from generate_prompt import *
from stage_2_function import *

import glom.grouping
import requests
from glom import core

# ==========================================
# 测试用例示例（需要在环境中安装对应的包）
# ==========================================



    

if __name__ == "__main__":

    print("正在清理未记录的API...")
    # api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")
    #clean_undocumented_apis_from_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")