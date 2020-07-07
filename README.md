# Exploring Automated Question Answering Methods for Teaching Assistance
## Authors: Brian Zylich, Adam Viola, Brokk Toggerson, Lara Al-Hariri, Andrew Lan

This repository contains source code related to this AIED 2020 paper: https://link.springer.com/chapter/10.1007/978-3-030-52237-7_49. 

If you have any questions, please contact me at bzylich@umass.edu.

## Requirements
- DrQA (installation instructions: https://github.com/facebookresearch/DrQA)
- CoreNLP (included in DrQA instructions)
- PyTorch
- PyTorch Lightning (https://github.com/PyTorchLightning/pytorch-lightning)
- Huggingface Transformers (https://github.com/huggingface/transformers)

## Pipeline (process documents through producing answers):

1. Preprocess docs
    - Build the json file from all of the individual documents: python parse_pdf.py  (make sure all file paths in this file are set correctly for your setup)

2. Prepare docs
    - Build database: python \<path-to-DrQA\>/DrQA/scripts/retriever/build_db.py \<path-to-data-in-JSON-format\>/docs.json \<output-path-db\>/docs.db
    - Build tf-idf features: python \<path-to-DrQA\>/DrQA/scripts/retriever/build_tfidf.py \<output-path-db\>/docs.db \<output-path-tf-idf\>    (tf-idf file name will be generated automatically)

3. Generate candidate answers:
    - python physics_drqa_two_step.py  (make sure all file paths in this file are set correctly for your setup)
    - this will generate a csv file with candidate answers (includes each document score, answer score, answerability score).

