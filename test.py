from config import *
from function import *
from main import *
from generate_prompt import *

# print(torch_samename_data["torch.where_1"])

# def log_analysis():
#     with open('conditions/ds_tf_log.txt', 'r', encoding='utf-8') as file:
#         log_list = [line.rstrip('\n') for line in file.readlines()]
#     for i in range(len(log_list)):
#         print(log_list[i])
#         if i > 10:
#             break   
# log_analysis()
with open(f"{lib_name}_APIdef.txt", 'r', encoding='utf-8') as file:
    api_defs = [line.strip() for line in file]

a = "0"
b = "torch.stack"
k = []
if a == "0":
    api_names = read_file(f"{lib_name}_APIdef.txt")

    for i in range(len(api_names)):

        doc = get_doc(api_names[i])
        local_add_log(api_defs[i])
        local_add_log(doc)
        i+=1
        local_add_log(str(i) + '/' + str(len(api_names)))
        if doc is not None and doc is not False and len(doc)<250:
            k.append(i)
    local_add_log(k)
elif a == "1":
    print(get_doc(b))
else:
    print(eval(a).__doc__)


[11,32,39,]
