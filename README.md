# LLM Scale Consistency Evaluation

This repository contains the dataset, evaluation results, and analysis
code for the master's thesis:

> **A Framework for Consistent Multiple-Choice Question Answering with
> Large Language Models across Varying Scales**
>
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
│   └── consistency_results_<strategy>_<mode>.csv  # 10 of 13 configs (see note)
├── figures/                                    # 16 thesis figures (PNG)
├── code/
│   ├── curation/          # Dataset construction pipeline (Scripts 1–3 + runner)
│   ├── evaluation/        # 13 inference scripts (one per strategy–mode config)
│   └── analysis/          # APD calculation, dataset statistics, figure generation
└── scripts/
    └── slurm/             # HPC job scripts (LRZ AI cluster)
```

---

## Data

The dataset comprises 1,494 unique questions each presented in five
answer-scale variants (7,470 evaluation items total), drawn from:

| Source | Waves / Year | Questions |
|--------|-------------|-----------|
| World Values Survey (WVS) | Waves 4–7 (1999–2022) | 171 |
| European Values Study (EVS) | 2017 | 69 |
| Understanding Society | Wave 1 (2009–2010) | 19 |
| OpinionQA (Santurkar et al., 2023) | 15 Pew survey waves | 1,235 |
| **Total** | | **1,494** |

**Source PDFs** (WVS, EVS, Understanding Society) are not included due
to redistribution restrictions. They are publicly available from their
respective data archives.

---

## Results

10 of the 13 results files are included directly in `results/`. The three
Chain-of-Thought configuration files (`cot_zeroshot`, `cot_oneshot`,
`cot_chainwise`) are excluded because the reasoning text column causes
them to exceed GitHub's 100 MB file size limit. They are available at:

> https://drive.google.com/drive/folders/1zUBLpRf-QpZibeD9DORv_q-wxKGgsiZg?usp=drive_link

---

## Reproducing the Dataset

Run the curation pipeline in sequence:

```bash
cd code/curation
python run_pipeline.py
```

This executes three scripts in order:
1. `curate_wvs_evs.py` — extracts and tidies WVS/EVS questions
2. `add_understanding_society.py` — appends the 19 Understanding Society items
3. `finalize_dataset.py` — generates five scale variants per question and
   combines with OpinionQA

The source PDFs must be placed in `raw_pdfs/` before running. The OpinionQA
source file (`opinionQA_questions_final.csv`) must be placed in
`extracted_content/` before running Script 3.

---

## Running Evaluation

Each script in `code/evaluation/` corresponds to one strategy–mode
configuration. Example:

```bash
python evaluate_consistency_vanilla_zeroshot.py
```

Evaluation was conducted on the LRZ AI cluster (NVIDIA H100 GPUs).
SLURM job scripts are provided in `scripts/slurm/` for reference; paths
and partition parameters are specific to the LRZ environment.

---

## Analysis and Figures

```bash
# Dataset statistics (Chapter 3)
python code/analysis/dataset_statistics.py

# APD calculation (Chapter 6 & 7)
python code/analysis/calculate_apd.py --input results/consistency_results_vanilla_zeroshot.csv \
                                       --output results/apd_vanilla_zeroshot.csv

# All thesis figures (Chapters 5, 6 & 7)
python code/analysis/generate_figures.py
```

Helper functions shared across analysis scripts are in `code/analysis/utils.py`.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
pdfplumber
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn pdfplumber
```

---

## Citation

If you use this dataset or code, please cite:

```bibtex
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