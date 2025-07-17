from config import *
from function import *
from main import *
from generate_prompt import *




# 尝试加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 尝试加载模型
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)