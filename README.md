# A Contrastive Framework for Neural Text Generation
**Authors**: Harsh Raj, Domenic Rosati, Subhabrata Majumdar

This repository contains code, data, and other related resources of our paper ["Measuring Reliability of Large Language Models through Semantic Consistency"](https://arxiv.org/abs/2211.05853).

****

### Files
- models.py - contains all the paraphrasing methods wrapped into classes.
- pert_input.py - generates paraphrases for a text dataset. (used [TruthfulQA](https://huggingface.co/datasets/truthful_qa))
- generate_answers-TruthfulQA.ipynb - uses a models (OPT-series, GPT3 (text-davinci-002)) to generate outputs for sentences and their paraphrases.
- compute_consistency_score-TruthfulQA.ipynb - computes consistency score for questions and paraphrases (with outputs gebnerated from all the models used).

### Data
- truthful_qa_top_6_by_pp.csv - top 6 paraphrases generated for randomly sampled questions from [TruthfulQA](https://huggingface.co/datasets/truthful_qa) dataset.