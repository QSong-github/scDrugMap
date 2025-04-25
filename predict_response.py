import argparse
import os
import subprocess

model_dirs = {
    "Geneformer": "./Geneformer-finetuing-lora-prompt_cell_cls",
    "tGPT": "./tGPT-main/tGPT-main/tGPT-main",
    "UCE": "./UCE-main/UCE-main",
    "scBERT": "./scBERT-master/scBERT-master",
    "CellPLM": "./CellPLM-main/CellPLM-main",
    "CellLM": "./OpenBioMed-main/OpenBioMed-main",
    "scGPT": "./scGPT-main/scGPT-main/scgpt/tasks",
    "scFoundation": "./scFoundation-main/scFoundation-main/model",
}

def run_script(script):
    print(f">>> Running: {script}")
    subprocess.run(["python", script], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to run")
    args = parser.parse_args()
    model = args.model

    if model not in model_dirs:
        raise ValueError(f"Unsupported model: {model}. Choose from: {list(model_dirs.keys())}")

    work_dir = model_dirs[model]
    os.chdir(work_dir)
    print(f"=== Entering directory: {work_dir} ===")

    # Conditionally run dataset creation if script exists
    if os.path.exists("dataset_making.py"):
        run_script("dataset_making.py")

    run_script("get_ebd.py")
    run_script("benchmarking_main_EBD.py")
    run_script("benchmarking_main_FT.py")

    print(f"Finished processing for model: {model}")

if __name__ == "__main__":
    main()
