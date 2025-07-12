from config import *
from function import *
from main import *
from generate_prompt import *

all_args = ["input", "weight", "bias", "stride", "padding", "dilation", "groups"]


all_combinations = generate_all_combinations(all_args)

#for i in all_combinations:
#    print(i)

#print("————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")

resonal_combinations = [
    ["input", "weight"],
    ["input", "weight", "bias"],
    ["input", "weight", "stride"],
    ["input", "weight", "padding"],
    ["input", "weight", "dilation"],
    ["input", "weight", "groups"],
    ["input", "weight", "bias", "stride"],
    ["input", "weight", "bias", "padding"],
    ["input", "weight", "bias", "dilation"],
    ["input", "weight", "bias", "groups"],
    ["input", "weight", "stride", "padding"],
    ["input", "weight", "stride", "dilation"],
    ["input", "weight", "stride", "groups"],
    ["input", "weight", "padding", "dilation"],
    ["input", "weight", "padding", "groups"],
    ["input", "weight", "dilation", "groups"],
    ["input", "weight", "bias", "stride", "padding"],
    ["input", "weight", "bias", "stride", "dilation"],
    ["input", "weight", "bias", "stride", "groups"],
    ["input", "weight", "bias", "padding", "dilation"],
    ["input", "weight", "bias", "padding", "groups"],
    ["input", "weight", "bias", "dilation", "groups"],
    ["input", "weight", "stride", "padding", "dilation"],
    ["input", "weight", "stride", "padding", "groups"],
    ["input", "weight", "stride", "dilation", "groups"],
    ["input", "weight", "padding", "dilation", "groups"],
    ["input", "weight", "bias", "stride", "padding", "dilation"],
    ["input", "weight", "bias", "stride", "padding", "groups"],
    ["input", "weight", "bias", "stride", "dilation", "groups"],
    ["input", "weight", "bias", "padding", "dilation", "groups"],
    ["input", "weight", "stride", "padding", "dilation", "groups"],
    ["input", "weight", "bias", "stride", "padding", "dilation", "groups"]
]


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 测试  generate_all_combinations
# 
# 将列表转换为元组以便放入集合中
all_combinationsa_as_tuples = set(tuple(x) for x in all_combinations)
resonal_combinations_as_tuples = set(tuple(x) for x in resonal_combinations)
# 找到a中不在b中的元素
a_unique = all_combinationsa_as_tuples - resonal_combinations_as_tuples

# 找到b中不在a中的元素
b_unique = resonal_combinations_as_tuples - all_combinationsa_as_tuples

# 输出结果
#print("all_combinations中不在resonal_combinations中的元素：", [list(x) for x in a_unique])
#print("resonal_combinations中不在all_combinations中的元素：", [list(x) for x in b_unique])

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#测试  filter_combinations
# conditions = {
#     "Required Parameters": ["input", "weight"],
#     "Forced Coexistence": [],
#     "Mutual Exclusion": []
# }

#filtered_combinations = filter_combinations(all_combinations, conditions)
#for i in filtered_combinations:
#    print(i)

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

#if filtered_combinations == resonal_combinations:
#    print("good")




# 使用示例




