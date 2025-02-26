# scDrugMap

### This resource is an implementation of ''Fine-tuned large gene language model to predict drug sensitivity''. 


## Overview



##  Usage
We provide framework codes for 8 benchmark single-cell language models and 2 large language models for reference. We strongly recommend using our code-free [Webserver](https://scdrugmap.com/)



### Running
   Due to conflicting packages in the environments of various models, we cannot provide a unified running environment. Please refer to the original installation instructions of each project: [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [tGPT](https://github.com/deeplearningplus/tGPT), [UCE](https://github.com/snap-stanford/uce), [scBERT](https://github.com/TencentAILabHealthcare/scBERT), [CellPLM](https://github.com/OmicsML/CellPLM), [OpenBioMed/CellLM](https://github.com/PharMolix/OpenBioMed), [scGPT](https://github.com/bowang-lab/scGPT), [scFoundation](https://github.com/biomap-research/scFoundation). The complete code of the project is in [zenodo](https://scdrugmap.com/).
   
   (1) [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
   ```bash
   # entering the code directory
   $ cd ./benchmark/Geneformer-finetuing-lora-prompt_cell_cls/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```
   
   (2) [tGPT](https://github.com/deeplearningplus/tGPT)
   ```bash
   # entering the code directory
   $ cd ./benchmark/tGPT-main/tGPT-main/tGPT-main/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```

   (3) [UCE](https://github.com/snap-stanford/uce)
   ```bash
   # entering the code directory
   $ cd ./benchmark/UCE-main/UCE-main/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```
   
   (4) [scBERT](https://github.com/TencentAILabHealthcare/scBERT)
   ```bash
   # entering the code directory
   $ cd ./benchmark/scBERT-master/scBERT-master/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```
   
   (5) [CellPLM](https://github.com/OmicsML/CellPLM)
   ```bash
   # entering the code directory
   $ cd ./benchmark/CellPLM-main/CellPLM-main/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```

   (6) [OpenBioMed/CellLM](https://github.com/PharMolix/OpenBioMed)
   ```bash
   # entering the code directory
   $ cd ./benchmark/OpenBioMed-main/OpenBioMed-main/
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```

   (7) [scGPT](https://github.com/bowang-lab/scGPT)
   ```bash
   # entering the code directory
   $ cd ./benchmark/scGPT-main/scGPT-main/scgpt/tasks/
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```

   (8) [scFoundation](https://github.com/biomap-research/scFoundation)
   ```bash
   # entering the code directory
   $ cd ./benchmark/scFoundation-main/scFoundation-main/model/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_ebd.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   # train and test by fine-tuning with LOAR
   $ python benchmarking_main_FT.py
   ```

   (9) [Llama](https://github.com/meta-llama/llama3)
   ```bash
   # entering the code directory
   $ cd ./benchmark/Llama/
   # creating dataset
   $ python dataset_making.py
   # generating fixed embedding
   $ python get_embeds.py
   # train and test by fixed embedding
   $ python benchmarking_main_EBD.py
   ```

   (10) [GPT4-mini](https://openai.com/index/gpt-4/)
   ```bash
   # entering the code directory
   $ cd ./benchmark/GPT4/
   # prediction
   $ python main.py
   ```


## Dataset



## Reference
```
[1].Shen, H. et al. Generative pretraining from large-scale transcriptomes: Implications for single-cell deciphering and clinical translation. bioRxiv, 2022.2001. 2031.478596 (2022).
[2].Yang, F. et al. scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. Nature Machine Intelligence 4, 852-866 (2022).
[3].Theodoris, C.V. et al. Transfer learning enables predictions in network biology. Nature 618, 616-624 (2023).
[4].Zhao, S., Zhang, J. & Nie, Z. Large-scale cell representation learning via divide-and-conquer contrastive learning. arXiv preprint arXiv:2306.04371 (2023).
[5].Hao, M. et al. Large-scale foundation model on single-cell transcriptomics. Nature Methods, 1-11 (2024).
[6].Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods, 1-11 (2024).
[7].Wen, H. et al. CellPLM: pre-training of cell language model beyond single cells. bioRxiv, 2023.2010. 2003.560734 (2023).
[8].Rosen, Y. et al. Universal cell embeddings: A foundation model for cell biology. bioRxiv, 2023.2011. 2028.568918 (2023).
[9].AI@Meta Llama 3 Model Card. https://huggingface.co/meta-llama/Meta-Llama-3-8B (2024).
[10].OpenAI GPT-4o mini. https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/ (2024).
```
