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
    # model = AutoModelForCausalLM.from_pretrained(model_path,device_map={"": gpu_ids[0]} )
    # model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map={"": gpu_ids[0]} )
    # model = Starcoder2ForCausalLM.from_pretrained(model_path, device_map={"": gpu_ids[0]} )
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )
    
    i = 0

    while(True):
        # 获取函数名
        
        fun_string = api_names[i]
        i += 1
        # 获取函数文档字符串
        api_doc = get_doc(fun_string)
        if api_doc == False:
            add_log(f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
            continue
        # 生成prompt
        prompt_1 = generate_prompt_1(fun_string, api_doc)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 常见做法
        inputs = generate_input(prompt_1, tokenizer, model)

        # 把inputs放到模型参数所在设备
        inputs = inputs.to(next(model.parameters()).device)

        outputs = generate_output(inputs, model, tokenizer)
        # 解码输出
        outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        add_log("模型输出：\n" + outputs_text + "\n ______________________________________________________________________________________________________________________")
        
        api_conditions = handle_output(outputs_text, model_path)

        #存储至json
        append_api_condition_to_json(f'/tmp/Momo_test/{lib_name}_conditions.json', fun_string, api_conditions)
        add_log(f"已完成{fun_string}的API条件生成, 进度"+str(i)+"/"+str(len(api_names)))

        #if i >= len(api_names):
        if i >= 50:
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



