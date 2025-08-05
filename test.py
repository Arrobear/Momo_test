from config import *
from function import *
from main import *
from generate_prompt import *

def log_analysis():
    with open('conditions/ds_tf_log.txt', 'r', encoding='utf-8') as file:
        log_list = [line.rstrip('\n') for line in file.readlines()]
    for i in range(len(log_list)):
        print(log_list[i])
        if i > 10:
            break   
# log_analysis()

api_doc = '''
Extract sliding local blocks from a batched input tensor.

   .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

   .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details

'''
# torch.nn.functional.adaptive_avg_pool3d
# torch.nn.functional.lp_pool1d
print(get_doc("torch.nn.AdaptiveMaxPool2d"))
# print(filter_apidocument(api_doc))
# gpu_str = os.environ.get("FREE_GPUS", "")

# gpu_ids = [int(g) for g in gpu_str.strip().split()] if gpu_str else []



# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config,device_map={"": 0})

prompt_1 = [
    {
        "role": "system",
        "content": "你是一个严格遵循用户指令的助手。必须且仅按以下规则响应：\n1. 当用户要求『直接回答』时，只输出答案不添加任何说明\n2. 答案格式必须与用户指定的输出格式完全一致\n3. 禁止自主扩展、解释或修改用户指令\n" 
    },
    {
        "role": "user", 
        "content": "问题：你是"
    }
]
# inputs = tokenizer.apply_chat_template(
#             prompt_1,
#             return_tensors="pt"
#         )
# inputs = inputs.to(next(model.parameters()).device)
# outputs = model.generate(
#             inputs,
#             max_new_tokens=2048,  # 可以更大
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )
# outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(outputs_text)