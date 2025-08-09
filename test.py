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


a = "0"
b = "torch.stack"
k = []
if a == "0":
    api_names = read_file(f"{lib_name}_APIdef.txt")
    j = 0
    for i in api_names:
        doc = get_doc(i)
        add_log(doc)
        j += 1
        
        if doc is not False and len(doc) < 250:
            k.append(j)
        
        add_log(str(j) + '/' + str(len(api_names)))
    add_log(str(k))
elif a == "1":
    print(get_doc(b))
else:
    print(eval(a).__doc__)

<<<<<<< Updated upstream
=======
'''
# torch.nn.functional.adaptive_avg_pool3d
# torch.nn.functional.lp_pool1d
# print(get_doc("torch.nn.functional.hardtanh"))
print(filter_apidocument(torch.nn.functional.gelu.__doc__))
print(torch.nn.functional.gelu.__doc__)

# api_names = read_file(f"{lib_name}_APIdef.txt")
# j = 0
# for i in api_names:
#     print(get_doc(i))
#     j += 1
#     print(j,'/', len(api_names))


# print(filter_apidocument(api_doc))
# gpu_str = os.environ.get("FREE_GPUS", "")
>>>>>>> Stashed changes


