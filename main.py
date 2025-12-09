from config import *
from stage_1_function import *
from generate_prompt import *
from stage_1_approch import *


if __name__ == "__main__":
  print("", end = "")

  #读取函数文档``
  api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")
  print(f"共{len(api_names)}个API")

  # 生成过滤条件 → conditions
  #generate_api_conditions(api_names)
   
  # 生成所有可能的组合 → arg_combinations
  # base_condition_filter(api_names)

  # 调用大模型检查参数组合是否合法 → error_combinations
  # check_condition_filter(large_combination_list)

  # 过滤不合法组合得到最终参数组合 → 在generate_api_input中完成

  # 生成参数空间(在stage_2_function.py中完成) → arg_spaces

  # 获取源代码(在stage_2_function.py中完成) → api_src_code

  # 综合参数组合+源代码+文档字符串，生成输入
  # generate_api_boundary(api_names)

  # 根据上一步生成的boundary生成输入
  generate_api_input(api_names)

  # 生成测试案例模板
  # generate_test_cases(api_names)
  
  # 注入测试输入并运行
  # run_test_cases()


