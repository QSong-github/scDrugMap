# DrugTuner

## This resource is an implementation of ''Fine-tuned large gene language model to predict drug sensitivity'' (DrugTuner). Our papers are published at http://. . .

## The steps for running the code are as follows:

### 1.Installation Environment
#### This project is all completed with python code. Required dependencies are in 'requirements.txt', please create the python environment before running.

### 2.Download models and data
#### Due to the large size of the pre-trained model and original data, we do not provide it directly here. We used a model pre-trained by Geneformer [1], and it can be downloaded from 'Hugging Face' (https://huggingface.co/ctheodoris/Geneformer/tree/main) or github repository (https://github.com/jkobject/geneformer). The original data can be downloaded from google drive ().

### 3.Dataset construction
#### Execute python 'dt_rebuild.py' on the command line to build the data set. Note that we have provided the relevant dictionary files here. If you want to build it from scratch, you can refer to the 'dict.py' file and related materials (https://github.com/cx0/geneformer-finetune).

### 4.Running
#### To run the model, just execute 'python main.py' on the command line. But be careful to adjust the initialization parameters to suit your environment. By default, we provide a data set containing 800 pieces of samples as an example.

## Reference
### [1].Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., ... & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. Nature, 1-9.
