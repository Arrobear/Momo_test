from config import *
from stage_1_function import *
from generate_prompt import *
'''
存储整个方法中的小步骤 

generate_api_conditions(lib_name, api_names): 根据库名称和API名称生成API条件，并存储至JSON文件

'''



def generate_api_conditions(api_names):
    with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
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
    with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
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
    with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]

    large_combination_api = []

    i = 0   #ｉ：循环变量
    j = 0   #ｊ：json文件编号


    while True:
        # 读取json文件，每次读取一个函数的合理参数组合数组(若过大，则分批读取)
        error_combinations = []

        # 遍历每个函数的组合，检查是否满足条件
        fun_string = api_names[i]
        for def_ in api_defs:
            if fun_string in def_:
                api_def = def_
                break
          
        function_name = filter_samenames(i, fun_string, api_names)
        i += 1

        
        arg_combinations, j = get_all_combinations_from_json(function_name, j)
        api_doc = get_doc(function_name)
        if api_doc == False:
            add_log(f'/tmp/Momo_test/error_combinations/{lib_name}_log.txt',f"[错误] 获取 {fun_string} 的文档失败，跳过该函数")
            continue

        # last_combination = ['learning_rate', 'initial_accumulator_value', 'l1_regularization_strength', 'name', 'l2_shrinkage_regularization_strength', 'weight_decay', 'clipnorm', 'clipvalue', 'ema_momentum', 'ema_overwrite_frequency', 'loss_scale_factor', 'gradient_accumulation_steps']
        # last_key = False
        
        add_log(f'/tmp/Momo_test/error_combinations/{lib_name}_log_{j}.txt',f"准备检查 {function_name} 的参数组合，共 {len(arg_combinations)} 组, 当前函数文件编号 = {j}")
        n = 0  # 进度计数

        for arg_combination in arg_combinations:
        #     if last_key == False:
        #         if arg_combination == last_combination:
        #             last_key = True
        #             continue
        #         continue

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
                
                add_log(f'/tmp/Momo_test/error_combinations/{lib_name}_log_{j}.txt', f"[错误] {function_name} 的参数组合 {arg_combination} 可能不合法，已记录"+f"函数文件编号 = {j}")
                add_log(f'/tmp/Momo_test/error_combinations/{lib_name}_log_{j}.txt', "模型输出：\n" + outputs_text + "\n ______________________________________________________________________________________________________________________")
            
            n += 1
            print("当前进度："+str(n)+"/"+str(len(arg_combinations)))
        add_log(f'/tmp/Momo_test/error_combinations/{lib_name}_log_{j}.txt', f" {function_name} 的参数组合已确认，当前函数文件编号 = {j}")

        path = f'/tmp/Momo_test/error_combinations/error_{lib_name}_combinations.json'  # 非法参数组合文件路径
        append_filtered_combinations_to_json(path, function_name, error_combinations)


        if i >= len(api_names):
        # if i >= 1:
            break

    # add_log(f'/tmp/Momo_test/error_combinations/{lib_name}_log_{j}.txt', f"以下函数因参数过多，可能导致组合过大，未进行检查：{large_combination_api}")



#------------------------------------
# 生成api boundary
#------------------------------------

def generate_api_boundary(api_names):

    with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]
    api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")

    # 加载LLM模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )


    if lib_name == "torch":
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        j = 0
        path = f'/home/chaoni/haoyahui/documentation/api_input/{lib_name}_inputs_{j}.json'
        length_api_names = len(api_names)
        for i in range(length_api_names):
            
            api_inputs = []
            api_name = api_names[i]
            arg_combinations = read_json_api(api_name=api_name, file_path=f"../documentation/arg_combinations/", read_mode="combination")
            api_code = read_json_api(api_name=api_name, file_path=f"../documentation/api_src_code/", read_mode="src_code")
            error_combinations = read_json_api(api_name=api_name, file_path=f"../documentation/error_combinations/", read_mode="error_combination")
            conditions = read_json_api(api_name=api_name, file_path=f"../documentation/conditions/", read_mode="conditions")
            arg_spaces = read_json_api(api_name=api_name, file_path=f"../documentation/arg_space/", read_mode="arg_space")
            if error_combinations is None:
                error_combinations = []
            if arg_spaces is None:
                add_log("/home/chaoni/haoyahui/Momo_test/",api_name)
            length_arg_combinations = len(arg_combinations)

            for arg_combination in arg_combinations:

                if arg_combination in error_combinations:
                    continue
                else:
                    length_arg_spaces = len(arg_spaces)
                    if arg_spaces:
                        for arg_space in arg_spaces:
                            print("第"+str(i+1)+"/"+str(length_api_names)+"个API"+api_name+"，第"+ 
                                str(arg_combinations.index(arg_combination) + 1)+"/"+ str(length_arg_combinations)+"个参数组合，第"+ 
                                str(1+arg_spaces.index(arg_space))+"/"+ str(length_arg_spaces)+"个参数空间")
                            path_type = arg_space["path_type"]
                            prompt = generate_prompt_3(api_name, arg_combination, api_code, arg_space, conditions["Parameter type"])
                            if tokenizer.pad_token is None:
                                tokenizer.pad_token = tokenizer.eos_token  
                            inputs = generate_input(prompt, tokenizer, model)

                            # 把inputs放到模型参数所在设备
                            inputs = inputs.to(next(model.parameters()).device)

                            outputs = generate_output(inputs, model, tokenizer)
                            # 解码输出
                            outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            # print(outputs_text)
                            api_boundary = extract_clean_json(outputs_text)

                            # 根据api_input_boundary生成测试输入
                            # print("________________________________________________________________")
                            print(api_boundary)
                            # 将api_boundary转换为字典形式
                            # api_boundary = json.loads(api_boundary_str)
                            # api_input = generate_test_inputs_from_api_boundaries(api_name, api_boundary, model, tokenizer)
                            

                            new_api_input_boundary = {"path_type": path_type, "api_input": api_boundary}
                            api_inputs.append(new_api_input_boundary)
                    

            #存储至json
            if is_file_too_large(path, max_size_mb=1000):
                j+=1
                path = f'/home/chaoni/haoyahui/documentation/arg_boundary/{lib_name}_boundary_{j}.json'
                save_api_inputs(api_name, api_inputs, path)
            else:
                save_api_inputs(api_name, api_inputs, path)
            print(f"已完成{api_name}的API boundary生成, 进度"+str(i)+"/"+str(len(api_names)))
            if i == 50:
                break


    elif lib_name == "tf":
        pass
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        # prompt = generate_prompt_3(api_names)
        # 将输入存入json文件

    # 添加新的深度学习库
    else:
        pass

    return

#------------------------------------
# 生成api input
#------------------------------------

def generate_api_input(api_names):
    # with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
    #     api_defs = [line.strip() for line in file]
    api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")

    # 加载LLM模型
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )

    if lib_name == "torch":
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        j = 0
        k = 0
        path = f'/home/chaoni/haoyahui/documentation/api_input/{lib_name}_inputs_{j}.json'
        length_api_names = len(api_names)
        for i in range(length_api_names):
            api_inputs = []
            api_name = api_names[i]
            api_boundarys = read_json_api(api_name=api_name, file_path=f"../documentation/arg_boundary/{lib_name}_boundary_{k}.json", read_mode="boundary")
            if api_boundarys == None:
                k += 1
                api_boundarys = read_json_api(api_name=api_name, file_path=f"../documentation/arg_boundary/{lib_name}_boundary_{k}.json", read_mode="boundary")

            # 将api_boundary转换为字典形式
            # api_boundary = json.loads(api_boundary_str)
            n = 0
            for api_boundary in api_boundarys:
                
                api_input = generate_test_inputs_from_api_boundaries(api_name, api_boundary["api_input"], model = None, tokenizer = None)
                new_api_input = {"path_type": api_boundary["path_type"], "api_input": api_input}
                api_inputs.append(new_api_input)
                if n == 0:
                    break
            #存储至json
            if is_file_too_large(path, max_size_mb=1000):
                j+=1
                path = f'/home/chaoni/haoyahui/documentation/api_input/{lib_name}_input_{j}.json'
                save_api_inputs(api_name, api_inputs, path)
            else:
                save_api_inputs(api_name, api_inputs, path)
            print(f"已完成{api_name}的API输入生成, 进度"+str(i)+"/"+str(len(api_names)))
            if i < 1:
                break


    elif lib_name == "tf":
        pass
        # 根据lib_name生成不同的输入
        # 生成prompt   调用generate_prompt_3, 定义于generate_prompt.py
        # prompt = generate_prompt_3(api_names)
        # 将输入存入json文件

    # 添加新的深度学习库
    else:
        pass

    return

#------------------------------------
# 生成测试案例model
#------------------------------------
def generate_test_cases(api_names):
    # 加载LLM模型
    with open(f"../documentation/{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
        api_defs = [line.strip() for line in file]
    api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")

    # 加载LLM模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map={"": gpu_ids[0]} )

    if lib_name == "torch":
        j = 0
        path = f'/tmp/Momo_test/{lib_name}_case_{j}.json'

        for i in range(len(api_names)):
            # 获取函数名
            api_name = api_names[i]
            
            # 获取函数文档字符串
            function_name = filter_samenames(i, api_name, api_names)
            api_def = api_defs[i]
            i += 1
            api_doc = get_doc(function_name)
            # 生成prompt
            prompt_5 = generate_prompt_5(api_name,api_def, api_doc)
            
            print(prompt_5)
            if i == 1:
                break
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # 常见做法
            inputs = generate_input(prompt_5, tokenizer, model)

            # 把inputs放到模型参数所在设备
            inputs = inputs.to(next(model.parameters()).device)

            outputs = generate_output(inputs, model, tokenizer)
            # 解码输出
            outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            case = handle_output(outputs_text, model_path)
            # 运行测试案例，输出结果
                        #存储至json
            if is_file_too_large(path, max_size_mb=1000):
                j+=1
                path = f'/tmp/Momo_test/{lib_name}_case_{j}.json'
                save_api_inputs(api_name, case, path)
            else:
                save_api_inputs(api_name, case, path)
            print(f"已完成{api_name}的API测试案例model生成, 进度"+str(i)+"/"+str(len(api_names)))


    elif lib_name == "tf":
        pass
        # 根据上一步生成的api_input生成测试案例
        
        # 运行测试案例，输出结果

    # 添加新的深度学习库
    else:
        pass

    return



# api_names = read_file(f"../documentation/{lib_name}_APIdef.txt")
# generate_test_cases(api_names)
#------------------------------------
# 对测试案例model注入测试输入并运行
#------------------------------------
def run_test_cases():

    def run_api(*args, **kwargs):
        """Auto-generated test template for torch.nn.Conv2d"""
        # extract input before class instantiation
        input_tensor = kwargs.pop("input", None)
        model = torch.nn.Conv2d(*args, **kwargs)
        output = model(input_tensor)
        return output
    
    test_inputs = [
    {
        "input": torch.randn(1, 3, 32, 32),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3
    },
    {
        "input": torch.randn(8, 512, 1024, 1024),
        "in_channels": 512,
        "out_channels": 1024,
        "kernel_size": 5,
        "padding": 2
    },
    {
        "input": torch.randn(1, 3, 64, 64),
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "groups": 4
    }
]

    execute_api_template(run_api, test_inputs)
    return



# def run_api(*args, **kwargs):
#     """Auto-generated test template for torch.nn.functional.conv1d"""
#     # call the API
#     output = torch.nn.functional.conv1d(*args, **kwargs)
#     return output
