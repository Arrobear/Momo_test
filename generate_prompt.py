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
    "Conditional Mutual Exclusion Parameters":[["para_1", "para_2","(para_1>1)&(para_2>1)"] ]
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
    "Conditional Mutual Exclusion Parameters": [["strides", "dilation","(strides>1)&(dilation>1)"] ]
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
        # prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{ori_prompt}\n</user>"
    else:
        ori_prompt_1 = ori_prompt + notion_2
        prompt = ori_prompt_1

    return prompt


def generate_prompt_2(fun_string, args, api_def, api_doc):

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

    \n(2) Parameter Combination
    Parameter combination refers to the reasonable composition of parameters when calling APIs.
    One parameter combination for this API is as follows:
    {args}
    
    \n---
    \n3. Your Tasks:
    Based on the explicit definition of the API and the API documentation, especially the "Args" part in the API documentation: 
    Determine whether the parameter combination provided in the 'Parameter Combination' section is a valid input for the given API.

    \n---
    \n4.Output Format:

    True or False

    \n---
    \n5.Examples:
    Output Examples: 

    True

    '''
    system_prompt = '''
        You are an assistant who strictly follows user instructions. Response must be made only according to the following rules:\n
        1. The answer format must be completely consistent with the output format specified by the user;\n
        2. Prohibit autonomous extension, interpretation, or modification of user instructions.\n
        '''
    prompt = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": ori_prompt}
        ]
    # prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{ori_prompt}\n</user>"
    return prompt



def generate_prompt_3(api_name, arg_combination, api_code, arg_space, conditions):
    #补充api文档中的条件用于生成输入
    py_code = api_code["python"]["code"]
    cpp_code = api_code["cpp"]["code"]
    arg_path = arg_space["conjuncts"]
    arg_intro = {}
    for arg in arg_combination:
        if arg in conditions:
            arg_intro[arg] = conditions[arg]
        
    ori_prompt = f'''
    \n1. Role:
        You are an expert in [{lib_name}], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

    \n---
    \n2. Background and Context:

    \nWe provide below the official documentation for the API to be analyzed: 
    [{api_name}]

    \n(1) Parameter Combination
    Parameter combination refers to the reasonable composition of parameters when calling APIs.
    The parameter combination with constraints for this API is as follows:
    {arg_intro}
    
    \n(2) Parameter space
    The parameter space consists of the control flow key code of the function, which is a constraint representing the execution path of the API.
    A parameter space is shown as follows:
    {arg_path}

    \n(3) Function source code
    The function source code includes extractable Python encapsulation layer and CPP implementation layer code.
    Source Code is as follows:
    Python:
    {py_code}
    CPP:
    {cpp_code}

    \n---
    \n3. Your Tasks:
    Based on the three conditions of Parameter Combination, Parameter space, and Function source code above,Your task is to generate a normalized parameter space schema describing how each argument of the API behaves.

    Your responsibilities:
    Focus only on the parameters in the Parameter Combination (from Parameter Combination).

    Determine parameter types (Tensor, int, float, bool, str, or optional).

    For Tensor parameters: infer the minimum and maximum shape values (shape_min, shape_max) based on provided examples or common conventions.

    For string parameters (str):Identify possible categorical values (choices) such as "mean", "sum", "none", "reflect", "zeros", etc.

    For scalar parameters (int, float, bool): Provide their valid value range (min, max) or possible values if enumerable.
    
    Infer possible dtypes (e.g. torch.float32, torch.float64, torch.complex64).

    Detect logical constraints between parameters, such as:shape consistency (input.shape[1] == weight.shape[1])、dtype consistency (input.dtype == weight.dtype)、divisibility (Cin % groups == 0)、ordering or boundary rules (kernel_size <= input_length)

    Output only factual, structured information.
    Do not invent nonexistent parameters.
    If something is uncertain, use a reasonable but concise estimate (e.g. [1, 1024] for lengths).

    Your output must capture the essential constraints and boundaries necessary for generating test inputs, not the actual test cases.
    \n---
    \n4.Output Format:
    {"{"}
    "params": {"{"}
        "<param_name>": {"{"}
        "type": "<Tensor|int|float|bool|str|optional|other>",
        "shape_min": [ ... ],        // for Tensor only
        "shape_max": [ ... ],        // for Tensor only
        "dtypes": [ ... ],           // for Tensor only
        "min": <number>,             // for numeric parameters
        "max": <number>,
        "choices": [ ... ]           // for string or enum-like parameters
        {"}"},
        ...
    {"}"},
    "constraints": [
        "<logical_expression_1>",
        "<logical_expression_2>",
        ...
    ]
    {"}"}
    Rules:
        Only generate the parameters in the Parameter Combination
        Output must be valid JSON and syntactically correct.
        Include all relevant fields for each parameter type.
        Do not invent extra fields or comments.
        For unknown values, infer conservative defaults (e.g. "choices": ["default"]).

    \n---
    \n5.Examples:
    Output Examples: 
    {"{"}
    "params": {"{"}
        "input": {"{"}
        "type": "Tensor",
        "shape_min": [1, 1, 1],
        "shape_max": [8, 16, 1024],
        "dtypes": ["torch.float32", "torch.float64", "torch.complex64"]
        {"}"},
        "pad": {"{"}
        "type": "tuple[int]",
        "min": 0,
        "max": 10
        {"}"},
        "mode": {"{"}
        "type": "str",
        "choices": ["constant", "reflect", "replicate", "circular"]
        {"}"},
        "value": {"{"}
        "type": "float",
        "min": -1e6,
        "max": 1e6
        {"}"}
    {"}"},
    "constraints": [
        "mode == 'constant' or value == 0.0",
        "len(pad) % 2 == 0",
        "input.dtype in ['torch.float32', 'torch.float64']"
    ]
    {"}"}



    '''
    system_prompt = '''
        You are an assistant who strictly follows user instructions. Response must be made only according to the following rules:\n
        1. The answer format must be completely consistent with the output format specified by the user;\n
        2. Prohibit autonomous extension, interpretation, or modification of user instructions.\n
        '''
    prompt = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": ori_prompt}
        ]
    # prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{ori_prompt}\n</user>"
    return prompt


def generate_prompt_4(api_name, param_name, param_info, param_constraints, api_doc):

    ori_prompt = f'''
    1. Role:
        You are an expert in deep learning frameworks and libraries, with deep knowledge of API design, 
        parameter semantics, configuration structures, constraint reasoning, and practical usage patterns.
        You specialize in generating **valid, diverse, constraint-satisfying test values** for complex parameters 
        such as config objects or nested option dictionaries.

    ---
    2. Background and Context:
        You will generate **multiple legal test values for one single complex parameter** of an API.
        Here are some API information for reference:
        • API name:
            {api_name}

        • Parameter to generate:
            {param_name}

        • Parameter information (schema, structure, allowed types):
            {param_info}

        • Parameter constraints (hard rules that MUST be satisfied):
            {param_constraints}

            These constraints may include:
              - required or optional fields
              - allowed ranges, enums, shapes, dtypes
              - inter-field relationships
              - mutually exclusive fields
              - conditional requirements (if A then B must be X)
              - boundary semantics
              - hidden assumptions

        • API documentation:
            {api_doc}

          You should use these to infer additional rules, semantics, default behaviors, modes, 
          and hidden constraints that are not explicitly listed.

        • Important:
            You only need to generate values for **this one complex parameter**.
            Any other simple API parameters (int/float/tensor) are not part of the generation task.

    ---
    3. Your Tasks:

        Your primary task is:
            Based on the provided parameter structure (param_info), parameter constraints (param_constraints), 
            API documentation (api_doc), 
            generate multiple (at least two) valid and diverse test values for the parameter "{param_name}".

            Each generated test value must:
            • Strictly satisfy all type, field, value-range, and structural requirements defined in param_info  
            • Strictly satisfy all hard constraints listed in param_constraints 
            (including cross-field relationships, conditional rules, mutual exclusivity, etc.)  
            • Comply with the semantic logic inferred from the API documentation and source code 
            (such as valid modes, default behaviors, field dependencies, etc.)  
            • Be a complete, directly usable value for automated testing  
            • Include at least two values with meaningful diversity 
            (e.g., different modes, boolean combinations, numeric ranges, boundary cases, etc.)

        ---

        Notes:

            • For simple fields (int/float/bool/string/enum), provide concrete explicit values.
            • Do NOT output actual tensor values in the JSON.
            • Always output tensor construction code as a string (e.g. "torch.randn(32, 32, dtype=torch.float32)").
            • At least one generated value should represent a "typical configuration" 
            and another should represent a "boundary/extreme configuration".
            • Generated values must be complete and fully parseable, without missing required fields 
            (unless the field is optional by definition).
            • No generated value may violate any explicit or implicit constraint.
            • The final output must be JSON only, with no explanations, comments, or additional text.


    ---
    4. Output Format:
        Your final output MUST be a valid JSON object with the structure:

            {{
                "test_values": [
                    <test_value_1>,
                    <test_value_2>,
                    ...
                ]
            }}

        Requirements:
        • JSON only — no comments, no additional explanation.
        • Each element inside "test_values" is the full value for the parameter "{param_name}".
        • The number of test values must be ≥ 2.
        • The JSON must be machine-parsable.

    ---
    5. Examples:
        (Example is illustrative and does NOT reflect the real parameter info.)

        Example output:

        {{
            "test_values": [
                {{
                    "mode": "fast",
                    "use_bias": true,
                    "dropout": 0.1,
                    "activation": "relu",
                    "norm": {{
                        "type": "layernorm",
                        "eps": 1e-5
                    }}
                }},
                {{
                    "mode": "accurate",
                    "use_bias": false,
                    "dropout": 0.0,
                    "activation": "gelu",
                    "norm": {{
                        "type": "batchnorm",
                        "momentum": 0.9,
                        "eps": 1e-3
                    }}
                }}
            ]
        }}

        Remember:
        • In the real task, strictly follow the schema in param_info and constraints in param_constraints.
        • Output ONLY the JSON object.
    '''

    system_prompt = '''
        You are an assistant who strictly follows user instructions. Response must be made only according to the following rules:\n
        1. The answer format must be completely consistent with the output format specified by the user;\n
        2. Prohibit autonomous extension, interpretation, or modification of user instructions.\n
        '''
    prompt = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": ori_prompt}
        ]
    # prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{ori_prompt}\n</user>"
    return prompt


def generate_prompt_5(api_name, arg_signature, api_doc):

    ori_prompt = f'''
    \n1. Role:
        You are an expert in [{lib_name}], deeply familiar with its API design and usage.including how classes (nn.Module) and functions handle their arguments.

    \n---
    \n2. Background and Context:

    \nThe following information describes the API you need to generate a test template for.
    API Name — This is the exact identifier of the API entry (e.g., 'torch.nn.Conv2d' or 'torch.fft.fft').
      It indicates whether the target is a class, function, or module method.

    Parameter Signature — This shows the expected arguments and their order as defined by the library.
      It tells you what parameters the API accepts and which ones might be optional.

    API Documentation — This is the official docstring or summary of the API’s purpose and behavior.
      Use it to infer whether the API should be instantiated (e.g., nn.Module) or called directly (e.g., torch function).
    Provided details:
     API Name: {api_name}
     Parameter Signature: {arg_signature}
     API Documentation:
      {api_doc}

    \n---
    \n3. Your Tasks:
        Generate a **minimal, safe, and runnable Python function** that can dynamically accept 
    multiple arguments and execute the API. 
    The function must:
    - Be named `run_api`.
    - Accept all external arguments through (*args, **kwargs).
    - If the API is a class (e.g., torch.nn.Conv2d):
        * Extract `input` (and other forward-only arguments) using `kwargs.pop()`
          before creating the instance, to avoid passing them into the constructor.
        * Instantiate the class, then call it with the popped inputs.
    - If the API is a function (e.g., torch.fft.fft), call it directly using all arguments.
    - Do not generate or modify any input tensors.
    - Do not print or explain anything — only output the runnable Python code.

    \n---
    \n4.Output Format:
    Output must follow this exact structure:

    ```python
    def run_api(*args, **kwargs):
        \"\"\"Auto-generated test template for {api_name}\"\"\"
        # prepare / instantiate if needed
        ...
        # call the API
        output = ...
        return output
    ```

    Replace `...` with appropriate logic based on API type.
    The function must be directly runnable once external arguments are injected.

---
    \n---
    \n5.Examples:
    Output Examples:

    Example for a **function-type API**:
    ```python
    def run_api(*args, **kwargs):
        \"\"\"Auto-generated test template for torch.fft.fft\"\"\"
        output = torch.fft.fft(*args, **kwargs)
        return output
    ```

    Example for a **class-type API**:
    ```python
    def run_api(*args, **kwargs):
        \"\"\"Auto-generated test template for torch.nn.Conv2d\"\"\"
        # extract input before class instantiation
        input_tensor = kwargs.pop("input", None)
        model = torch.nn.Conv2d(*args, **kwargs)
        output = model(input_tensor)
        return output
    ```
---
---

    '''
    system_prompt = '''
        You are an assistant who strictly follows user instructions. Response must be made only according to the following rules:\n
        1. The answer format must be completely consistent with the output format specified by the user;\n
        2. Prohibit autonomous extension, interpretation, or modification of user instructions.\n
        '''
    prompt = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": ori_prompt}
        ]
    # prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{ori_prompt}\n</user>"
    return prompt



def generate_prompt_6(combo, constraint):

    ori_prompt = f"""
1. Role:
    You are an expert in [{lib_name}] and Python static analysis. 
    You specialize in reasoning about tensor shapes, dtypes, devices, index semantics, and API parameter constraints
    *without running any code*.

---
2. Background and Context:

    (1) API parameters (a concrete combo of arguments):
    {combo}

    (2) Parameter constraints (a list of boolean conditions that must ALL hold):
    {constraint}

    Notes:
    - These constraints are written in a Python style (e.g. "input.dtype == index.dtype").
    - Identifiers in constraints (such as input, index, dim, etc.) refer to the parameters or objects defined in the combo.
    - Your job is to reason about whether, under standard [{lib_name}] and Python semantics, ALL of these constraints
      are satisfied by the given combo.

---
3. Your Tasks:

    Based on the given parameter combination (combo) and parameter constraints (constraints),
      determine whether the combo satisfies all constraints without executing any code. 
      If any constraint is clearly violated or if you cannot determine whether it is satisfied, the final result should be False. 
      Only when you are certain that all constraints are satisfied by the combo should you output True. 
      
      You must strictly follow the parameter semantics, tensor properties, and comparison relationships to make logical judgments, 
      and you must not provide explanations or reasoning steps—only output the final True or False according to your judgment.

---
4. Output Format (VERY IMPORTANT):

    - You must output ONLY one of the following two strings:
        True
        False

    - Do NOT output explanations, reasons, code, or any other text.
    - No extra spaces, no quotes, no punctuation, no newlines before or after.
    - If the combo satisfies all constraints under [{lib_name}] semantics → output True
      otherwise → output False.

---
5. Examples (for format only, NOT related to this question):

    Output Examples:

    True
    False

    Remember: for the actual question, your answer must be exactly one of these two.
    """
    system_prompt = '''
        You are an assistant who strictly follows user instructions. Response must be made only according to the following rules:\n
        1. The answer format must be completely consistent with the output format specified by the user;\n
        2. Prohibit autonomous extension, interpretation, or modification of user instructions.\n
        '''
    prompt = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": ori_prompt}
        ]
    # prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{ori_prompt}\n</user>"
    return prompt