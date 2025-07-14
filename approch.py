from config import *
from function import *
from generate_prompt import *
'''
存储整个方法中的小步骤

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''



def generate_api_conditions(api_names):
    #加载模型

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # to use Multiple GPUs do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    while(True):
        # 获取函数名
        i = 0
        fun_string = api_names[i]
        i += 1
        # 获取函数文档字符串
        api_doc = get_doc(fun_string)

        # 生成prompt
        prompt_1 = generate_prompt_1(fun_string, api_doc)

        # 调用LLM模型生成API条件
        inputs = tokenizer.encode(prompt_1, return_tensors="pt").to(device)
        outputs = model.generate(inputs)
        api_conditions = tokenizer.decode(outputs[0])

        #存储至json

        append_api_condition_to_json(f'{lib_name}_conditions.json', fun_string, api_conditions)
        print(f"已完成{fun_string}的API条件生成")

        if i >= 1:
            break

def base_condition_filter(api_names):
    while(True):
        # 获取函数名
        i = 0
        fun_string = api_names[i]
        i += 1

        # 获取函数文档字符串
        api_doc = get_doc(fun_string)
        
        # 选择对应的参数列表提取方法提取参数参数列表
        approach = f'extract_parameters_{lib_name}'
        apprameters_list = eval(approach)(api_doc)

        # 生成全参数组合
        all_combinations = generate_all_combinations(apprameters_list)
        # for j in all_combinations:
        #     print(j)   
        # 读取json得到过滤条件
        conditions = get_api_conditions(fun_string, f'{lib_name}_conditions.json')

        # 过滤参数组合
        filtered_combinations = filter_combinations(all_combinations, conditions)

        # 将过滤后的组合存储至json

        append_filtered_combinations_to_json(f'{lib_name}_combinations.json', fun_string, filtered_combinations)
        print(f"已完成{fun_string}的条件过滤")

        if i >= len(api_names):
            break



