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
    log_path = f'/tmp/Momo_test/{lib_name}_log.txt'
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
            add_log(log_path ,f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
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
        add_log(log_path, "模型输出：\n" + outputs_text + "\n ______________________________________________________________________________________________________________________")
        
        api_conditions = handle_output(outputs_text, model_path)

        #存储至json
        path = f'/tmp/Momo_test/{lib_name}_conditions.json'
        append_api_condition_to_json(path, function_name, api_conditions)
        add_log(log_path, f"已完成{function_name}的API条件生成, 进度"+str(i)+"/"+str(len(api_names)))

        if i >= len(api_names):
        # if i >= 50:
            break

def base_condition_filter(api_names):
    with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    i = 0
    log_path = f'/tmp/Momo_test/arg_combinations/{lib_name}_log.txt'
    # j: json文件编号
    j = 0
    
    while(True):
        # 获取函数名

        fun_string = api_names[i]
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
        path = f'/tmp/Momo_test/arg_combinations/{lib_name}_combinations_{j}.json'
        if os.path.exists(path):
            if is_file_too_large(path, max_size_mb=10):
                # 如果文件过大，换一个新文件
                # 确保目录存在
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # 创建空文件
                j += 1
                append_filtered_combinations_to_json(path, function_name, filtered_combinations)
            else:
                append_filtered_combinations_to_json(path, function_name, filtered_combinations)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump({}, f)  # 创建一个空的JSON文件
            append_filtered_combinations_to_json(path, function_name, filtered_combinations)

        
        add_log(log_path ,f"已完成{function_name}的条件过滤, 进度"+str(i)+"/"+str(len(api_names)))

        if i >= len(api_names):
            break

def check_condition_filter(api_names):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )
    with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    i = 119   #ｉ：循环变量
    j = 1   #ｊ：json文件编号

    log_path = f'/tmp/Momo_test/error_combinations/{lib_name}_log.txt'


    while True:
        # 读取json文件，每次读取一个函数的合理参数组合数组(若过大，则分批读取)
        error_combinations = []

        # 遍历每个函数的组合，检查是否满足条件
        fun_string = api_names[i]
        api_def = api_defs[i]   
        function_name = filter_samenames(i, fun_string, api_names)
        i += 1

        if fun_string == "tf.keras.optimizers.Ftrl":
            last_result = extract_invalid_parameter_combinations()
            for i in last_result:
                error_combinations.append(i)
                
        arg_combinations, j = get_all_combinations_from_json(function_name, j)
        api_doc = get_doc(function_name)
        if api_doc == False:
            add_log(f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
            continue

        
        for arg_combination in arg_combinations:

            # 输出（从json中删除）不满足条件的组合  fun_string, args, api_def, api_doc
            prompt_2 = generate_prompt_2(fun_string,arg_combination, api_def, api_doc)
        
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # 常见做法
            inputs = generate_input(prompt_2, tokenizer, model)

            # 把inputs放到模型参数所在设备
            inputs = inputs.to(next(model.parameters()).device)

            outputs = generate_output(inputs, model, tokenizer)
            # 解码输出
            outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            
            error_tag = handle_output(outputs_text, model_path)
            # add_log(log_path, error_tag)
            if 'False' in error_tag:
                error_combinations.append(arg_combination)
                
                add_log(log_path, f"[错误] {function_name} 的参数组合 {arg_combination} 可能不合法，已记录"+f"函数文件编号 = {j}")
                add_log(log_path, "模型输出：\n" + outputs_text + "\n ______________________________________________________________________________________________________________________")
        
        add_log(log_path, f" {function_name} 的参数组合已确认，当前函数文件编号 = {j}")

        path = f'/tmp/Momo_test/error_combinations/error_{lib_name}_combinations.json'  # 非法参数组合文件路径
        append_filtered_combinations_to_json(path, function_name, error_combinations)


        if i >= len(api_names):
        # if i >= 1:
            break

