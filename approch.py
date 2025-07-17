from config import *
from function import *
from generate_prompt import *
'''
存储整个方法中的小步骤

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''



def generate_api_conditions(api_names):
    #加载模型
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # 可选参数，控制量化精度
        llm_int8_has_fp16_weight=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # to use Multiple GPUs do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config,torch_dtype=torch.float16 ).to(device)
    i = 0

    while(True):
        # 获取函数名
        
        fun_string = api_names[i]
        i += 1
        # 获取函数文档字符串
        api_doc = get_doc(fun_string)

        # 生成prompt
        prompt_1 = generate_prompt_1(fun_string, api_doc)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 常见做法

        # 调用LLM模型生成API条件
        inputs = tokenizer(
            prompt_1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # 你可以根据模型设置合适长度
        ).to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,  # 明确设置
            eos_token_id=tokenizer.eos_token_id
        )
        api_conditions = tokenizer.decode(outputs[0], skip_special_tokens=True)

        #存储至json
        print(api_conditions)

        #append_api_condition_to_json(f'{lib_name}_conditions.json', fun_string, api_conditions)
        print(f"已完成{fun_string}的API条件生成")

        #if i >= len(api_names):
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



