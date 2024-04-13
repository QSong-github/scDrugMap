# DrugTuner

### This resource is an implementation of ''Fine-tuned large gene language model to predict drug sensitivity'' (DrugTuner). Our paper is released at [Here](http://。。。。).


## Overview



## The steps for running the code are as follows:

### 1.Installation Environment
This project is all completed with python code. Required dependencies are in ```./DrugTuner/requirements.txt```, please activate the python environment and run ```pip install -r requirements.txt``` to install environment.

### 2.Download models and data
Due to the large size of the pre-trained model and original data, we have not provide it directly here. We used the pre-trained model namely Geneformer [[1]](https://www.nature.com/articles/s41586-023-06139-9), and it can be downloaded from [Hugging Face](https://huggingface.co/ctheodoris/Geneformer/tree/main) or [github repository](https://github.com/jkobject/geneformer). We save the pre-trained model in ```./DrugTuner/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224```. Note that Geneformer provides two versions of pre-trained models including 12L and 6L. The original data can be downloaded from [google drive](https://). By default, the directory for original data is ```./DrugTuner/original_data```.

### 3.Dataset construction
Run ```python dt_rebuild.py``` on the command line to build the dataset. The dataset will be stored in the ```./DrugTuner/gene_cell_dt``` path. Here we provide a small dataset in the ```./DrugTuner/sub_gene_cell_dt``` containing 800 cells as a demo.

Note that we have provided the relevant dictionary files here. If you want to build it from scratch, you can refer to the ```dict.py``` file and related [materials](https://github.com/cx0/geneformer-finetune).

### 4.Running
Run ```python main.py``` on the command line to get the results. And be careful to adjust the initialization parameters to suit your conditions. 

## Reference
[1].Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., ... & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. Nature, 1-9.
