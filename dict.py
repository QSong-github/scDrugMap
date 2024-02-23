import pickle
import csv

# name_id_dict
with open('E:/Geneformer-finetuing-lora-prompt/dict_creation/gene_name_id_dict.pkl', 'rb') as file:
    name_id_dictionary = pickle.load(file)

swapped_dict = {v: k for k, v in name_id_dictionary.items()}

# token_id_dict
with open('E:/Geneformer-finetuing-lora-prompt/dict_creation/token_dictionary.pkl', 'rb') as file:
    token_dictionary = pickle.load(file)


# token_name_dict
new_dict = {}
# special token
new_dict.update({'<pad>': 0})
new_dict.update({'<mask>': 1})
new_dict.update({'A': 25426})
new_dict.update({'G': 25427})
new_dict.update({'C': 25428})
new_dict.update({'T': 25429})
new_dict.update({'<cell_cls>': 25430})
new_dict.update({'<gene_cls>': 25431})
new_dict.update({'<hybrid_cls>': 25432})
new_dict.update({'<sep>': 25433})
new_dict.update({'<prompt_beg>': 25434})
new_dict.update({'<prompt_end>': 25435})


for id in token_dictionary:
    if id in swapped_dict:
        new_dict[swapped_dict[id]]=token_dictionary[id]

print(new_dict)

# save
file_path = './dictionary_genename_token_pair.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(new_dict, file)

