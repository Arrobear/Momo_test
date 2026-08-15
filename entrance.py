from config import *
from stage_1_function import *
from generate_prompt import *
from stage_1_approch import *
# metamorphic_test 在需要时才 import，避免 Python 3.6 兼容性问题
# from metamorphic_test import run_metamorphic_tests


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--k", type=int, default=200,
                      help="Number of test cases per API (K)")
  parser.add_argument("--test-mode", choices=["auto", "v1", "v2"], default="auto",
                      help="Explicit differential-test mode; auto keeps legacy baseline detection")
  args = parser.parse_args()

  random.seed(42)
  #读取函数文档``
  api_names = read_file(f"../documentation/lib_api/{lib_name}_APIdef.txt")

  #print(f"共{len(api_names)}个API")

  # ==========================================
  # 阶段1: 准备阶段 (按需执行)
  # ==========================================

  # # 生成过滤条件 → conditions
  # generate_api_conditions(api_names)
  # print("generate_api_conditions")

  # # 生成所有可能的组合 → arg_combinations
  # base_condition_filter(api_names)
  # print("base_condition_filter")

  # # 调用大模型检查参数组合是否合法 → error_combinations
  # check_condition_filter(api_names)
  # print("check_condition_filter")

  # # 生成参数空间(在stage_2_function.py中完成) → arg_spaces

  # # 获取源代码(在stage_2_function.py中完成) → api_src_code

  # # 对生成的参数组合进行剪枝，得到最终的参数组合 → cut_combinations
  # cut_combinations(api_names)
  # print("cut_combinations")

  # # 生成输入边界 → arg_boundary
  # generate_api_boundary(api_names)
  # print("generate_api_boundary")

  # # 生成默认输入用于约束校验
  # generate_default_inputs(api_names)
  # print("generate_default_inputs")

  # # 根据boundary生成测试输入候选值
  # generate_api_input(api_names)
  # print("generate_api_input")

  # # 生成测试案例模板 (run_api 函数)
  # generate_test_cases(api_names)
  # print("generate_test_cases")

  # ==========================================
  # 阶段2: 测试执行和Bug发现
  # ==========================================

  # V1/V2 差分测试 (发现regression)
  run_test_cases(K=args.k, mode=args.test_mode)

  # run_test_cases(K=100)

  # 蜕变测试 (发现latent bug — 不依赖版本)
  # run_metamorphic_tests(num_samples=10)

