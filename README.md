# Chatbot Personas For QA Using Fine-Tuned LLMs

Fraud Detection Package with fine-tuned RoBERTa model, equipped with ethical considerations such as Differential Privacy and SMPC

- [Chatbot Personas For QA Using Fine-Tuned LLMs](#chatbot-personas-for-qa-using-fine-tuned-llms)
  - [Team](#team)
  - [CI/CD Pipeline Status](#cicd-pipeline-status)
  - [Installation and Run Instructions](#installation-and-run-instructions)
  - [Abstract](#abstract)
  - [Dataset Description](#dataset-description)
  - [Training Methodology](#training-methodology)
  - [Evaluation Methodology](#evaluation-methodology)
  - [References](#references)
  - [Citations](#citations)


***

## Team
|Members|
|---|
|Advaith Shyamsunder Rao|
|Falgun Malhotra|
|Vanshita Gupta|

***

## CI/CD Pipeline Status

[![Persona LLM Chatbot CI/CD](https://github.com/vanshigupta04/Persona-LLM-Chatbot-For-QA/actions/workflows/pipeline.yml/badge.svg?branch=main)](https://github.com/vanshigupta04/Persona-LLM-Chatbot-For-QA/actions/workflows/pipeline.yml)

***

## Installation and Run Instructions

**All helper functions and run steps can be found in the wiki pages.**

| Helper | Page |
| ------ | ------ |
| Setup Environment and Integrations | [Wiki](https://github.com/vanshigupta04/Persona-LLM-Chatbot-For-QA/wiki/Setup-Repository) |

***

## Abstract

The aim of this project is to create several Question Answering Chatbot personas using fine-tuned LLM models. As part of the model fine-tuning, PEFT techniques such as QLoRA and AdaLoRA will be devised for efficient and quicker weight updates. The performance of the persona LLM will be measured using MCQ as well as Free Response style evaluation, and benchmarked against the base LLM model.

***

## Dataset Description

The initial phase of the project focuses on building a QA Chatbot for the persona **#1 Friends Fan**. This persona will be a bot well versed with information about the TV show Friends (1994-2004). For this persona the [ConvoKit Friends corpus](https://convokit.cornell.edu/documentation/friends.html) will be used. Across the 10 seasons there are 236 episodes, 3,107 scenes (conversations), 67,373 utterances, and 700 characters (users). More Personas such as **Family Doctor** and **Jack Sparrow** will be explored through the project.

The raw corpus of text for each persona, will be put through the data preparation pipeline, which will fetch data from respective sources, perform necessary data preprocessing and cleaning. This will ensure that the data pull and formatting are consistent, enabling the project to scale well across more persona-additions.

For each persona, an evaluation dataset will be created. The evaluation dataset will contain the question, reference answer, and 3 wrong answers.

***

## Training Methodology

To effectively run large language models (LLMs) on devices with limited processing power, compressed models like GGML or GGUF will be employed. GPT4all or llama.cpp executables will be utilized to incorporate both CPU and GPU, ensuring quick response times for inference tasks. For each persona, the underlying LLM model will be fine-tuned on cloud computing systems using parameter-efficient optimization techniques such as QLoRA and AdaLoRA.

***

## Evaluation Methodology

The following evaluation metrics will be used as part of the project. A curated evaluation dataset will be used to validate each persona chatbot.

1. **MCQ Evaluation:** Defines the ratio of correct answers by the model to the total questions in the dataset. 

2. **Free Response Evaluation:** Defines n-gram overlap score between chatbot response and gold output. The metric will be a weighted average of scores such as Bleu, Rouge, Meteor, and chrF.

3. **Correct format answer rate:** Defines how many times the LLM model response matches a specific MCQ-output format. This metric will be used to validate optimal prompt engineering for MCQ style evaluation.

For MCQ evaluation, prompt engineering techniques such as Few-shot prompting will be used to ensure the LLM response matches a specific format of answer such as (A), (B), (C), or (D).

To comprehensively evaluate the Persona LLM, the following evaluation will be performed
Benchmark the performance of Persona LLM(fine-tuned) vs Base LLM using the metrics (1) and (2) above.

For the model prompt in MCQ evaluation metric, measure the metric (3) above for experiments around prompt engineering. Experiments include using zero-shot and few-shot prompting and [microsoft guidance](https://github.com/guidance-ai/guidance).

***

## References

1. MMLU Evaluation
   - **URL**: [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)

2. QLoRA
   - **URL**: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

3. ADALoRA
   - **URL**: [https://arxiv.org/abs/2303.10512](https://arxiv.org/abs/2303.10512)

4. Friends Show Conversation Corpus
   - **URL**: [https://convokit.cornell.edu/documentation/friends.html](https://convokit.cornell.edu/documentation/friends.html)

5. NGram Overlap Metrics
   - **URL**: [https://medium.com/explorations-in-language-and-learning/metrics-for-nlg-evaluation-c89b6a781054](https://medium.com/explorations-in-language-and-learning/metrics-for-nlg-evaluation-c89b6a781054)


## Citations

**`Friends Conversation Corpus Dataset`**

- **Title**: Character-Mining
- **URL**: [https://github.com/emorynlp/character-mining](https://github.com/emorynlp/character-mining)
- **Copyright**: Copyright 2015,2016,2017,2018,2019,2020 Emory University