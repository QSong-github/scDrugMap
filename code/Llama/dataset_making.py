import csv
import os
import pickle
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("../llama3")
directory_path = "/home/exouser/llama2/lxndt_filter/"
csv.field_size_limit(50000000)

def rebuilder(directory_path):
    files_list = os.listdir(directory_path)

    for filename in files_list:
        if os.path.exists('./samples/' +  filename[:-4] + '_samples.pkl'):
            print(filename[:-4] + ' finished')
            continue
        csv_file_path = directory_path + filename
        print(csv_file_path, 'processing...')
        headr = True
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = []
            labels = []
            samples= []
            # sequence level
            #for row in csv_reader:
            for row in tqdm(csv_reader, ncols=80, position=0, leave=True):
                row_data = row[0].split('\t')
                if headr:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        pattern.append(gene)
                    headr = False
                    # print(pattern)
                else:
                    assert len(pattern)==len(row_data)
                    bioNLP_seq = ''
                    # token level
                    for i in range(len(row_data)):
                        if i==0:
                            pass
                        elif i==1:
                            if 'sensitive' in row_data[i]:
                                labels.append(1)
                            elif 'resistant' in row_data[i]:
                                labels.append(0)
                        else:
                            if row_data[i]==0:
                                pass
                            else:
                                bioNLP_seq = bioNLP_seq + str(pattern[i]) + str(row_data[i])+' '

                    seq_token = tokenizer(bioNLP_seq, max_length=4096, truncation=True)
                    sample = seq_token

                    samples.append(sample['input_ids'])


        # dataset save
        dt_pkl = {}
        dt_pkl['input_ids']= samples
        dt_pkl['labels']= labels
        with open('./samples/' +  filename[:-4] + '_samples.pkl', 'wb') as PKLfile:
            pickle.dump(dt_pkl, PKLfile)

        print('saved ' + filename)


rebuilder(directory_path)




