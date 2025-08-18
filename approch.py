from config import *
from function import *
from generate_prompt import *
'''
存储整个方法中的小步骤 

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''



def generate_api_conditions(api_names):
    with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path,device_map={"": gpu_ids[0]} )
    # model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map={"": gpu_ids[0]} )
    # model = Starcoder2ForCausalLM.from_pretrained(model_path, device_map={"": gpu_ids[0]} )
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )

    i = 0

    while(True):
        # 获取函数名
        fun_string = api_names[i]
        api_def = api_defs[i]
        
        # 获取函数文档字符串
        function_name = filter_samenames(i, fun_string, api_names)
        i += 1
        api_doc = get_doc(function_name)
        if api_doc == False:
            add_log(f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
            continue

        # 生成prompt
        prompt_1 = generate_prompt_1(fun_string, api_def, api_doc)
        
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
        append_api_condition_to_json(f'/tmp/Momo_test/{lib_name}_conditions.json', function_name, api_conditions)
        add_log(f"已完成{function_name}的API条件生成, 进度"+str(i)+"/"+str(len(api_names)))

        if i >= len(api_names):
        # if i >= 50:
            break

def base_condition_filter(api_names):
    with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    i = 0

    while(True):
        # 获取函数名

        fun_string = api_names[i]
        api_def = api_defs[i]
        function_name = filter_samenames(i, fun_string, api_names)

        # 得到所有合法参数→生成所有组合→过滤合法组合→存储至json
        args = get_all_parameters(function_name)
        # 生成全参数组合
        all_combinations = generate_all_combinations(args)

        i += 1
        # for j in all_combinations:
        #     print(j)   
        # 读取json得到过滤条件
        json_path = Path(__file__).parent / "conditions" / f"{lib_name}_conditions.json"

        conditions = get_api_conditions(function_name, str(json_path))

        # 过滤参数组合
        filtered_combinations = filter_combinations(all_combinations, conditions)

        # 将过滤后的组合存储至json

        append_filtered_combinations_to_json(f'/tmp/Momo_test/{lib_name}_combinations_6.json', function_name, filtered_combinations)
        add_log(f"已完成{function_name}的条件过滤, 进度"+str(i)+"/"+str(len(api_names)))

        if i >= len(api_names):
            break



