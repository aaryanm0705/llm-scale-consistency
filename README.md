# LLM Scale Consistency Evaluation

This repository contains the dataset, evaluation results, and analysis 
code for the master's thesis:

> **A Framework for Consistent Multiple-Choice Question Answering with 
> Large Language Models across Varying Scales**  
> Aaryan Mallayanmath, Ludwig-Maximilians-Universität München, 2026

---

## Overview

This work evaluates the response consistency of six instruction-tuned 
language models when the same survey-style multiple-choice question is 
presented with different but semantically equivalent answer scale labels. 
Consistency is measured using Average Pairwise Distance (APD) across 
13 prompting configurations.

**Models evaluated:** Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, 
Qwen2.5-7B-Instruct, gemma-2-9b-it, deepseek-llm-7b-chat, glm-4-9b-chat-hf

**Prompting strategies:** Vanilla, Expert, Inconsistent, Chain-of-Thought, 
Scale-Anchored

**Application modes:** Zero-shot, One-shot, Chainwise

---

## Repository Structure
```
repo-root/
├── data/
│   ├── combined_dataset.csv                    # 7,470 evaluation items
│   ├── combined_original_source_questions.csv  # 1,494 questions at question level
│   └── curation_log.csv                        # 38-entry manual exclusion log
├── results/
│   └── consistency_results_<strategy>_<mode>.csv  # 10 of 13 configs
├── code/
│   ├── curation/          # Dataset construction pipeline
│   ├── evaluation/        # 13 inference scripts
│   └── analysis/          # APD calculation and figure generation
└── scripts/
    └── slurm/             # HPC job scripts (LRZ AI cluster)
```

---

## Data

The dataset comprises 1,494 unique questions each presented in five 
answer-scale variants (7,470 evaluation items total), drawn from:

- World Values Survey (WVS), waves 4–7
- European Values Study (EVS), 2017
- Understanding Society, Wave 1 (UK Data Archive Study 6614)
- OpinionQA benchmark (Santurkar et al., 2023)

**Source PDFs** (WVS, EVS, Understanding Society) are not included due 
to redistribution restrictions. They are publicly available from their 
respective data archives.

---

## Results

10 of 13 results files are included directly. The three CoT configuration 
files are excluded as they exceed GitHub's 100 MB file size limit due to 
the reasoning text column. They are available on request.

---

## Reproducing the Dataset

Run the curation pipeline in sequence:
```bash
cd code/curation
python run_pipeline.py
```

Requires the source PDFs to be placed in the appropriate input directory 
before running.

---

## Running Evaluation

Each script in `code/evaluation/` corresponds to one strategy--mode 
configuration. Example:
```bash
python evaluate_consistency_vanilla_zeroshot.py
```

Evaluation was conducted on the LRZ AI cluster (NVIDIA H100 GPUs). 
SLURM job scripts are provided in `scripts/slurm/` for reference; 
paths and partition parameters are specific to the LRZ environment.

---

## Analysis and Figures
```bash
# Dataset statistics (Chapter 3)
python code/analysis/dataset_statistics.py

# APD calculation (Chapter 6 & 7)
python code/analysis/apd_calculation.py

# Figure generation (Chapters 5 & 6)
python code/analysis/generate_figures.py
```

---

## Citation

If you use this dataset or code, please cite:
```
@mastersthesis{mallayanmath2026,
  author  = {Mallayanmath, Aaryan},
  title   = {A Framework for Consistent Multiple-Choice Question Answering 
             with Large Language Models across Varying Scales},
  school  = {Ludwig-Maximilians-Universität München},
  year    = {2026}
}
```

---

## Collaborative Component

The OpinionQA scale-variant adaptation used in this dataset was developed 
in the complementary bachelor's thesis:

> Kandlinger, M. (2026). *Consistency of LLMs in Multiple-Choice Question 
> Answering: An Empirical Evaluation across Varying Scales.* 
> Ludwig-Maximilians-Universität München.