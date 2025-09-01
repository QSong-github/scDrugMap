# scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction

### Overview
scDrugMap is an integrated Python toolkit and interactive web server designed for predicting drug response at single-cell resolution using large-scale foundation models (FMs). This framework provides a comprehensive benchmarking platform to evaluate model performance across diverse cancer types, therapy classes, tissue types, and drug regimens.
In addition to robust performance evaluation, scDrugMap supports various training strategies, including layer freezing, fine-tuning via LoRA, and zero-shot inference, and includes tools for biomarker discovery, model interpretability, and drug resistance analysis.[[paper](https://arxiv.org/abs/2505.05612)]

### Key Features
- **🧬 Support for 10 Foundation Models**:
  - 8 single-cell foundation models (e.g., scFoundation, scGPT, tGPT)
  - 2 general-purpose LLMs (LLaMa3-8B, GPT4o-mini)

- **⚙️ Multi-strategy Model Training**:
  - Layer-freezing
  - Fine-tuning with LoRA (Low-Rank Adaptation)
  - Zero-shot/few-shot prediction

- **📊 Two Evaluation Strategies**:
  - Pooled-data evaluation: Train/test on aggregated datasets
  - Cross-data evaluation: Test generalizability across distinct datasets

- **📁 Rich and Curated Dataset Repository**:
  - 326,751 single cells (primary collection)
  - 18,856 single cells (validation collection)
  - 14 cancer types, 5 tissue types, 3 therapy types, and 21 regimens

- **🌐 Web Interface for Easy Access**:
  - Explore models, datasets, predictions, and visualizations via https://scdrugmap.com



### Web Server 🚀
The web server enables users to: Upload scRNA-seq data; Select models and evaluation settings; Predict drug response at single-cell resolution; Visualize results interactively

🔗 Visit: https://scdrugmap.com

### Installation
Clone the repository: `git clone https://github.com/QSong-github/scDrugMap.git`

Set up your environment: `conda env create -f env.yaml`

You can also follow the original environment installation instructions for every model.
[Geneformer](https://huggingface.co/ctheodoris/Geneformer), [tGPT](https://github.com/deeplearningplus/tGPT), [UCE](https://github.com/snap-stanford/uce), [scBERT](https://github.com/TencentAILabHealthcare/scBERT), [CellPLM](https://github.com/OmicsML/CellPLM), [OpenBioMed/CellLM](https://github.com/PharMolix/OpenBioMed), [scGPT](https://github.com/bowang-lab/scGPT), [scFoundation](https://github.com/biomap-research/scFoundation). The complete code of the project is in [zenodo](https://zenodo.org/records/14938211).


### Docker Deployment 🐳

For quick deployment using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/QSong-github/scDrugMap.git
cd scDrugMap

# Start the services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```
### Getting Started
   (0) One-click launch
   ```bash
   # entering the project directory
   $ cd ./benchmark/
   # Specify a model (e.g. scFoundation, scGPT) and training mode (EBD and FT)
   $ python launcher.py --model scGPT --mode EBD
   ```

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
   # train and test by fine-tuning with LORA
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
   # train and test by fine-tuning with LORA
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
   # train and test by fine-tuning with LORA
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
   # train and test by fine-tuning with LORA
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
   # train and test by fine-tuning with LORA
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
   # train and test by fine-tuning with LORA
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
   # train and test by fine-tuning with LORA
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

### Reference
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

**Important Notes:**
- The backend image contains multiple LLM weights and is quite large (~10GB+), so the initial pull may take some time depending on your internet connection
- The frontend will be available at: http://localhost:3000
- The backend API will be available at: http://localhost:8000
- Both services will start automatically and the frontend will wait for the backend to be ready


### Citation
@article{wang2025scdrugmap,
  title={scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction},
  author={Wang, Qing and Pan, Yining and Zhou, Minghao and Tang, Zijia and Wang, Yanfei and Wang, Guangyu and Song, Qianqian},
  journal={arXiv preprint arXiv:2505.05612},
  year={2025}
}
### License
MIT License © Qianqian Song Lab
