from config import *
from function import *
from generate_prompt import *
from approch import *


if __name__ == "__main__":
  print("", end = "")

  #读取函数文档``
  api_names = read_file(f"{lib_name}_APIdef.txt")

  # 生成过滤条件
  #generate_api_conditions(api_names)
   
  # 生成所有可能的组合
  # base_condition_filter(api_names)

  # 调用大模型检查参数组合是否合法
  check_condition_filter(large_combination_list)

  '''
    补充实验代码
    写过滤函数
    补充实验 
  '''

    #编写条件生成和过滤部分代码
    #链接服务器测试

    # 获取并打印文档字符串
    # for i in range(3):
    #     API_doc = get_doc(fun_string[0][i])
    #     print("_____________________________________________________________________    __________________________________________________________________________________")
    #     #print(API_doc)

    #     print(extract_parameters_torch(API_doc))

    # for i in range(3):
    #     API_doc = get_doc(fun_string[1][i])
    #     print("_____________________________________________________________________    __________________________________________________________________________________")
    #     #print(API_doc)

    #     print(extract_parameters_tf(API_doc))
