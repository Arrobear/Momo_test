from config import *
from function import *
from main import *
from generate_prompt import *

a = get_doc("torch.normal")
print(a)
approach = f'extract_parameters_{lib_name}'
apprameters_list = eval(approach)(a)
print(apprameters_list)