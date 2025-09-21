# CONQUER: CONTEXT-AWARE REPRESENTATION WITH QUERY ENHANCEMENT FOR TEXT-BASED PERSON SEARCH

## Introduction

This repository contains the PyTorch implementation for the paper [CONQUER: CONTEXT-AWARE REPRESENTATION WITH QUERY ENHANCEMENT FOR TEXT-BASED PERSON SEARCH]. Our work introduces a two-stage framework designed to address the challenges of cross-modal discrepancies and ambiguous user queries in Text-Based Person Search.

**Official Source Code**: [https://github.com/zqxie77/CONQUER](https://github.com/zqxie77/CONQUER) [cite: 11]

### News!

* **[2025-09-20]** Code and pre-trained models have been released!

### CONQUER Framework

Unlike existing methods that perform a direct search using the original text query, the CONQUER framework improves the query at inference time without needing to retrain the backbone model. The process begins by finding a relevant anchor image. A Multimodal Large Language Model (MLLM) then learns key visual details from this image through a Q&A process. Finally, these details are fused with the original text to create an improved query that is used to re-rank the search results. This is all supported by the training phase, where the Context-Aware Representation Enhancement (CARE) module learns robust cross-modal embeddings.

## Requirements and Datasets

* PyTorch
* OpenAI CLIP ViT-B/16 (Image Encoder)
* CLIP Transformer (Text Encoder) 
* Qwen2.5-VL-7B (for IQE module) 

### Datasets

We evaluate our model on three widely-used TBPS benchmarks

**CUHK-PEDES**.
**ICFG-PEDES**.
**RSTPReid**.

## Training and Evaluation

### Stage 1: Training the CARE Module

To train a new CONQUER model from scratch, run the following script. [cite_start]This stage trains the Context-Aware Representation Enhancement (CARE) module to learn robust cross-modal embeddings[cite: 37].

```bash
sh run_CONQUER.sh
```
### Stage 2: Inference with the IQE Module

To perform inference and evaluate a trained model, run the following script. This stage uses the plug-and-play Interactive Query Enhancement (IQE) module to refine queries and improve retrieval results.
```bash
sh run_IQE.sh
```
