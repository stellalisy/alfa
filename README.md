# ALFA: Aligning LLMs to Ask Good Questions

This repository implements the **ALFA** framework for improving large language models’ ability to ask high-quality follow-up questions in clinical reasoning scenarios. It includes code for:

- **Data processing** (preparing real-world interactions from r/AskDocs)  
- **Counterfactual data generation** (synthesizing diverse question variations with specific attributes)  
- **Preference-based optimization** via DPO/PPO/RLHF  
- **Evaluation** on both single-turn question quality and an **interactive clinical reasoning** benchmark (MediQ-AskDocs)

Below is an overview of the repository structure and pointers on how to run various components. For detailed technical explanations and design decisions, please see the associated paper and supplementary documentation.

---

## Table of Contents
- [Repository Structure](#repository-structure)  
- [Installation & Environment](#installation--environment)  
- [Data Preparation](#data-preparation)  
- [Counterfactual Generation](#counterfactual-generation)  
- [Preference Modeling & Training](#preference-modeling--training)  
- [MediQ Evaluation & Benchmarking](#mediq-evaluation--benchmarking)  
- [Ranking & Human Evaluation](#ranking--human-evaluation)  
- [How to Run](#how-to-run)  
- [Citation & Acknowledgments](#citation--acknowledgments)

---


## Key Directories

- **data/**  
  Holds raw and processed data for training and evaluation (r/AskDocs data, prompts, ID lists).  
  - **ids/**: Files listing specific train/test/eval question IDs.  
  - **mediq_eval/**: Data for the interactive MediQ experiments, including conversation files.  
  - **prompts/**: Prompt templates and references for LLM data generation.

- **src/**  
  Source code for data processing, counterfactual generation, evaluation, etc.  
  - **counterfactual_generation/**: Scripts to synthesize "enhanced" or "corrupted" question variants for clarity, relevance, answerability, etc.  
  - **data_loader/**: Scripts to create preference training files, supervised fine-tuning (SFT) data, test splits, etc.  
  - **mediq_eval/**: Code for running interactive clinical QA with the MediQ framework.
  - **rank_eval/**: Tools for pairwise ranking of generated questions (LLM-based or human annotator).
  - **training/**: Contains RLHF code (OpenRLHF) and pipelines for DPO/PPO or reward modeling.
    - **sample_configs/**: Example YAML config files for each training/fine-tuning stage.
    - **scripts/**: Stand-alone scripts to run various tasks (DPO, PPO, SFT, merging model weights, or generating questions in batch).
---

## 0. Installation & Environment

To reproduce the paper results, you would need to follow each step, but for the ALFA framework and evaluation, skip to Step 3. 
1. **Clone the Repo**  
```bash
git clone https://github.com/stellalisy/alfa.git
cd alfa
```

2. **Set Up Conda Environment**  
```bash
conda env create -f environment.yml
conda activate alfa
```

3. **Directory Permissions**  
Ensure you have appropriate read/write permissions for data/model checkpoints.

---

## 1. Data Preparation

1. **Prepare Raw Data**  
   Place your original r/AskDocs data in data/. The code in src/data_loader/ will expect certain file naming conventions.

2. **Generating Train/Test Splits**  
   Use scripts like create_sft_files.py or create_test_files.py to generate final .jsonl files for each split.

3. **Additional Metadata**  
   If you have labels or specialized contexts, put them in data/ids/.

---

## 2. Counterfactual Generation

Scripts in src/counterfactual_generation/ use an LLM to rewrite questions with different attributes.

- **generate.py**: Main script for attribute-based rewriting.  
- **verifier_filter.py**: Uses an LLM-based judge to confirm whether the generated rewrites match the intended direction.

```bash
cd src/counterfactual_generation
python generate.py --config path_to_generation_config.yaml
python verifier_filter.py --config path_to_verification_config.yaml
```

The output typically contains enhanced, original, and corrupted question versions in JSON.

---

## 3. The ALFA Framework

### Reward Model Training
Train a reward model to score question pairs as "better" or "worse."

```bash
python scripts/launch_rm_with_yaml.py --config sample_configs/sample_config_rm.yaml
```

### DPO or PPO Alignment
Use DPO (Direct Preference Optimization) or PPO. DPO is simpler, while PPO is RL-based.

```bash
# DPO
python scripts/launch_dpo_with_yaml.py --config sample_configs/sample_config_dpo.yaml
# PPO
python scripts/launch_ppo_with_yaml.py --config sample_configs/sample_config_ppo.yaml
```

### Supervised Fine-Tuning (SFT)
If you want standard SFT on real or synthetic data:

```bash
python scripts/launch_sft_with_yaml.py --config sample_configs/sample_config_sft.yaml
```

---

## 4. MediQ Evaluation & Benchmarking

MediQ is in src/mediq_eval/. It simulates doctor-patient interactions with an LLM question generator.

1. **Data Conversion**  
   Use scripts like generate_questions_post.py to convert QA files to MediQ format.

2. **Run the Simulator**  
```bash
cd src/mediq_eval
python evaluate.py --model_checkpoint path/to/aligned_model
```

This measures question quality and final diagnostic accuracy.

---

## 5. Ranking & Human Evaluation

- **rank_eval/rank_eval.py**: Ranks pairs of questions automatically with GPT-4 or a local LLM.  
- **annotators/**: Tools for collecting human preferences.

```bash
cd rank_eval
python run_rank_eval.py --config sample_config.yaml
```

---

## How to Run

example_run.sh files are provided in the training, mediq_eval, and rank_eval directories.

---

## Citation & Acknowledgments

If you use this code or MediQ-AskDocs in your work, please cite our paper:

```
@inproceedings{li2025aligningllmstoaskgoodquestions,
  title={Aligning LLMs to Ask Good Questions: A Case Study in Clinical Reasoning},
  author={Li, Shuyue Stella and Mun, Jimin and Brahman, Faeze and Ilgen, Jonathan S. and Tsvetkov, Yulia and Sap, Maarten},
  booktitle={Arxiv},
  year={2025}
}
```

- Thanks to r/AskDocs for their publicly shared Q&A data.
- This project uses code from [OpenRLHF].
- See the paper for more technical details.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
