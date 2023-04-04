# Measuring-Reliability-of-LLMs

This repository contains code, data, and other related resources for the paper [Measuring Reliability of Large Language Models through Semantic Consistency](https://arxiv.org/abs/2211.05853), H. Raj, D. Rosati, and S. Majumdar, NeurIPS 2022 ML Safety Workshop.

***

## Files
- `models.py` contains all the paraphrasing methods wrapped into classes.
- `pert_input.py` generates paraphrases for a text dataset. (used [TruthfulQA](https://huggingface.co/datasets/truthful_qa))
- `generate_answers-TruthfulQA.ipynb` uses a models (OPT-series, GPT3 (`text-davinci-002`)) to generate outputs for sentences and their paraphrases.
- `compute_consistency_score-TruthfulQA.ipynb` computes consistency score for questions and paraphrases (with outputs gebnerated from all the models used).

## Data
- `data/truthfulQA_alldata.csv` - final set of paraphrases and original questions based on the [TruthfulQA](https://huggingface.co/datasets/truthful_qa) dataset.