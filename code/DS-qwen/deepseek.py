# prompt_template=('Please identif the sensitivity of cells to drugs based on what you have learned. I will provide you with the names of the 10 genes with the highest expression in each cell and the source for these cells. Each cell is listed in one row. Please give a judgment for each row of cells. If you think the cell is sensitive, answer 1. If you think it is resistant, answer 0.'
#                  'Source:GSE161801_PI'
#
#                  'CR2, CD24, FAS, CXCR3, CD1c '
#                  'KLK3, KRT8, KLK2, MSMB, ACPP, KLK1, KLK4 '
#                  'MMRN1, FLT4, RELN, CCL21, PROXl, LYVE1 '
#                  'TPSAB1, FCER1A, TPSB2, KIT, CD69, HDC '
#                  'ACTA2, MY01B, ACTA2, ANPEP, DES, MCAM, PDGFRB, CSPG4'
#
#                  'Please merge the results into [].')
from utils import Accuracy_score, F1_score, AUROC_score, Precision_score, Recall_score
import re
import os
import pickle
import csv


from ollama import chat
from ollama import ChatResponse


pre_prompt='Please use your knowledge to identify the sensitivity of the following cells to drugs. I will provide you with the names of the 10 genes with the highest expression in each cell and the source of the data set for these cells. A total of 10 cells are stored in the outer list, and the genes of each cell are stored in the inner list. If you think the cell is sensitive, answer 1, and if you think it is resistant, answer 0.'
examples=''
post_prompt=('Please merge the results of 10 cells into a format like this example: {res:1,0,1,1,1,0,0,1,0,0}. \n '
             'Please merge the results of 10 cells into a format like this example: {res:1,0,1,1,1,0,0,1,0,0}. \n '
             'Please merge the results of 10 cells into a format like this example: {res:1,0,1,1,1,0,0,1,0,0}. \n')

source_prompt='This is the data source:'
gene_prompt='The following are the gene names:'


data_path = './samples/'
chunk_size=10


def extract_data_from_string(text):
    # 使用正则表达式查找大括号内以 "res:" 开头的数据
    pattern = re.compile(r'\{res:([0-1,]+)\}')
    match = pattern.search(text)

    if match:
        # 提取匹配的内容，并用逗号分割
        data_string = match.group(1)
        # 转换为列表并转为整数
        data_list = [int(num) for num in data_string.split(',')]
        return data_list
    else:
        return []

files_list_all = os.listdir(data_path)
for file in files_list_all:
    pred=[]
    output=[]
    with open(data_path+file, 'rb') as pklfile:
        PKLdata = pickle.load(pklfile)
        print('number:',len(PKLdata['labels']))
        source = file[:9]
        samples = PKLdata['samples']
        labels = PKLdata['labels']
        true=[]
        for i in range(0, len(samples), chunk_size):
            content = samples[i:i + chunk_size]
            lbl = labels[i:i + chunk_size]
            if len(content) != chunk_size:
                break

            input = pre_prompt + '\n\n' + source_prompt + '\n' + source + '\n\n' + gene_prompt + '\n' + str(content) + '\n\n' + post_prompt
            response: ChatResponse = chat(model='deepseek-r1:32B', messages=[{'role': 'user','content': input}])

            answer=str(response['message']['content'])

            predict = extract_data_from_string(answer)
            if len(predict) != 10:
                continue
            pred.extend(predict)
            true.extend(lbl)
            output.append(answer)

        print(len(pred))
        print(len(true))
        assert len(pred)==len(true)
        acc = Accuracy_score(pred, true)
        f1 = F1_score(pred, true)
        auroc = AUROC_score(pred, true)
        precision = Precision_score(pred, true)
        recall = Recall_score(pred, true)


        wr_data = [file[:-4], round(acc, 4), round(auroc, 4), round(precision, 4), round(recall, 4),round(f1, 4)]

        # 打开一个CSV文件，准备写入数据
        with open('./output/metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # 将列表作为一行写入CSV文件
            writer.writerow(wr_data)
        with open('./answer/chatgpt_logs_'+file[:-4]+'.pkl', "wb") as pklf:
            pickle.dump(output, pklf)
        print('finished:',file)


