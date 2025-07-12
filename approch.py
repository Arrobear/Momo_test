from config import *
from function import *
from generate_prompt import *
'''
存储整个方法中的小步骤

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''



def generate_api_conditions(lib_name, api_names):
    #加载模型

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # to use Multiple GPUs do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    while(True):
        # 获取函数名
        i = 0
        fun_string = api_names[i]
        i += 1
        # 获取函数文档字符串
        api_doc = get_doc(fun_string)

        # 生成prompt
        prompt_1 = generate_prompt_1(lib_name, fun_string, api_doc)

        # 调用LLM模型生成API条件
        inputs = tokenizer.encode(prompt_1, return_tensors="pt").to(device)
        outputs = model.generate(inputs)
        api_conditions = tokenizer.decode(outputs[0])

        #存储至json

        append_api_condition_to_json(fun_string, f'{lib_name}_conditions.json', api_doc)


        if i >= len(api_names):
            break