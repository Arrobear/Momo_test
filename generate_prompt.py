from config import *


'''
**该文件存储整个方法中生成prompt的函数**

generate_prompt_1(lib_name, fun_string, api_doc): 根据函数名和函数文档定制生成prompt_1



'''

#根据函数名和函数文档定制生成prompt_1
def generate_prompt_1(fun_string, api_def, api_doc):
    ori_prompt = f'''
    \n1. Role:
        You are an expert in [{lib_name}], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

    \n---
    \n2. Background and Context:

    \n(1) API Documentation.

    We provide below the official documentation for the API to be analyzed: 
    [{fun_string}]
    This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
    The explicit definition of the API is as follows:
    {api_def}

    The specific API documentation content is as below:
    [{api_doc}]

    \n(2) Parameter Dependency Types.

    In deep learning libraries such as [{lib_name}], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
    These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

    - "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
    - "Mutual Exclusion Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
    - "Conditional Mutual Exclusion Parameter Pairs" refer to parameter pairs that are not universally mutually exclusive but become incompatible only under specific conditions. In these cases, the mutual exclusion depends on certain value combinations, e.g., the parameters conflict only when both assume particular values or satisfy certain conditions based on their assigned values. Thus, these are a subset of "Mutual Exclusion Parameter Pairs" that exhibit conditional, rather than absolute, exclusivity.
    - "Mandatory coexistence parameters": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.
    - "Conditional Mutual Exclusion Parameters": When the parameters meet certain conditions, it will prevent the function from running.

    \n---
    \n3. Your Tasks:
    Based on the explicit definition of the API and the API documentation, especially the "Args" part in the API documentation: 

    1. Determine the Type of Each Parameter.
    For each explicitly defined parameter in [fun_string], determine its type, such as: tensor, int, str, optional tensor, etc.
    2. Identify Parameter Dependency Structures as accurately and completely as possible:
    - List all "Mandatory Parameters"  
    - Identify any "Mutual Exclusion Parameter Pairs"  
    - Identify any "Conditional Mutual Exclusion Parameter Pairs" 
    - Identify any "Mandatory Coexistence Parameters"

    \n---
    \n4.Output Format:
    
    {"{"}
    "Parameter type": {"{"}
        "input": "...",
        "weight": "...",
        "bias": "...",
        "stride": "...",
        "padding": "...",
        "dilation": "...",
        "groups": "..."
    {"}"},
    "Mandatory Parameters": ["...", "..."],
    "Mutually Exclusive Parameter Pairs": [["...", "..."], ...],
    "Mandatory Coexistence Parameters": [["...", "...", "..."], ["...", "..."], ...],
    "Conditional Mutual Exclusion Parameters":["para_1", "para_2","(para_1>1)&(para_2>1)"] 
    {"}"}

    \n---
    \n5.Examples:
    Output Examples: 
    {"{"}
    "Parameter type": {"{"}
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
    {"}"},
    "Mandatory Parameters": ["input", "weight"],
    "Mutually Exclusive Parameter Pairs": [],
    "Mandatory Coexistence Parameters": [],
    "Conditional Mutual Exclusion Parameters": ["strides", "dilation","(strides>1)&(dilation>1)"] 
    {"}"}
    '''

    notion_1 = '''
    \n6.Notions:
    Only output the json content of the example in the output format, do not add explanations.
    '''
    notion_2 = '''
    Please complete the corresponding information extraction based on the above content (output JSON directly):
    '''
    if model_path == "/nasdata/haoyahui/Model/Meta-Llama-3-70B-Instruct":
        ori_prompt_1 = ori_prompt + notion_1
        prompt = [
            {"role": "user", "content": ori_prompt_1}
        ]
    elif model_path  == "/nasdata/haoyahui/Model/DeepSeek-R1-Distill-Qwen-32B" :
        ori_prompt_1 = ori_prompt + notion_1
        system_prompt = '''
        You are an assistant who strictly follows user instructions. Response must be made only according to the following rules:\n
        1. The answer format must be completely consistent with the output format specified by the user;\n
        2. Prohibit autonomous extension, interpretation, or modification of user instructions.\n
        '''
        prompt = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": ori_prompt_1}
        ]
    else:
        ori_prompt_1 = ori_prompt + notion_2
        prompt = ori_prompt_1

    return prompt

