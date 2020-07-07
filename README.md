# Exploring Automated Question Answering Methods for Teaching Assistance
## Authors: Brian Zylich, Adam Viola, Brokk Toggerson, Lara Al-Hariri, Andrew Lan

This repository contains source code related to [our AIED 2020 paper](https://link.springer.com/chapter/10.1007/978-3-030-52237-7_49). 

If you have any questions, please contact me at bzylich@umass.edu.

## Requirements
- Python 3
- [DrQA](https://github.com/facebookresearch/DrQA)
- CoreNLP (included in DrQA instructions)
- PyTorch
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [Huggingface Transformers](https://github.com/huggingface/transformers)

## Pretrained answerability classifier
Our pretrained answerability classifier can be downloaded [here](https://drive.google.com/file/d/1zdBfffbYIDRdg7cqphz7vAKMDeJgfSCv/view?usp=sharing).

## Training the answerability classifier from scratch
Our answerability classifier was trained via 2 steps: training first on the [SQuAD 2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/) (using bert_lightning_adapter.py) and then finetuning on [Google's Natural Questions dataset](https://ai.google.com/research/NaturalQuestions) (using bert_lightning_adapter_transfer_nq.py). The Natural Questions dataset should be preprocessed using preprocess_nq.py.

## Pipeline (process documents through producing answers):

1. Preprocess docs
    - Build the json file from all of the individual documents: <pre><code>python parse_pdf.py  (make sure all file paths in this file are set correctly for your setup)</pre></code>

2. Prepare docs
    - Build database: <pre><code>python \<path-to-DrQA\>/DrQA/scripts/retriever/build_db.py \<path-to-data-in-JSON-format\>/docs.json \<output-path-db\>/docs.db</pre></code>
    - Build tf-idf features: <pre><code>python \<path-to-DrQA\>/DrQA/scripts/retriever/build_tfidf.py \<output-path-db\>/docs.db \<output-path-tf-idf\>    (tf-idf file name will be generated automatically)</pre></code>

3. Generate candidate answers:
    <pre><code>python physics_drqa_two_step.py  (make sure all file paths in this file are set correctly for your setup)</pre></code>
    - This will generate a csv file with candidate answers (includes each document score, answer score, answerability score)

