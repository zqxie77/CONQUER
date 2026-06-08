# CONQUER: CONTEXT-AWARE REPRESENTATION WITH QUERY ENHANCEMENT FOR TEXT-BASED PERSON SEARCH
[![IEEE](https://img.shields.io/badge/IEEE-11461248-0078C8.svg)](https://ieeexplore.ieee.org/document/11461248)
[![Project Page](https://img.shields.io/badge/GitHub-CONQUER-181717.svg)](https://github.com/zqxie77/CONQUER)

**Paper**: https://ieeexplore.ieee.org/document/11461248

## Introduction

This repository contains the PyTorch implementation for the paper [CONQUER: CONTEXT-AWARE REPRESENTATION WITH QUERY ENHANCEMENT FOR TEXT-BASED PERSON SEARCH]. Our work introduces a two-stage framework designed to address the challenges of cross-modal discrepancies and ambiguous user queries in Text-Based Person Search.

**Official Source Code**: [https://github.com/zqxie77/CONQUER](https://github.com/zqxie77/CONQUER)

### News!

* **[2026-01-25]** 🎉 **CONQUER** has been accepted by **ICASSP 2026**!
* **[2025-09-20]** Code and pre-trained models have been released.
 
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

To train a new CONQUER model from scratch, run the following script. This stage trains the Context-Aware Representation Enhancement (CARE) module to learn robust cross-modal embeddings.

```bash
sh run_CONQUER.sh
```
### Stage 2: Inference with the IQE Module

To perform inference and evaluate a trained model, run the following script. This stage uses the plug-and-play Interactive Query Enhancement (IQE) module to refine queries and improve retrieval results.
```bash
sh run_IQE.sh
```

### Citation

If you find this work useful in your research, please consider citing:

**BibTeX:**
```bash
@INPROCEEDINGS{11461248,
  author={Zeng, Chenxi and Duan, Yipeng and Xie, Zequn and Han, Xiaosong},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search}, 
  year={2026},
  volume={},
  number={},
  pages={12867-12871},
  keywords={Protocols;HTTP;LoRa;Local area networks;Videos;Communication systems;Video equipment;Data communication;Plugs;Fuses;Text-Based Person Search;Cross-modal Learning;Optimal Transport;Query Enhancement},
  doi={10.1109/ICASSP55912.2026.11461248}}
```

**RIS:**
```bash
TY  - CONF
TI  - CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search
T2  - ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
SP  - 12867
EP  - 12871
AU  - C. Zeng
AU  - Y. Duan
AU  - Z. Xie
AU  - X. Han
PY  - 2026
DO  - 10.1109/ICASSP55912.2026.11461248
JO  - ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
IS  - 
SN  - 2379-190X
VO  - 
VL  - 
JA  - ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
Y1  - 3-8 May 2026
ER  - 
```

**IEEE Style:**
C. Zeng, Y. Duan, Z. Xie and X. Han, "CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search," ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2026, pp. 12867-12871, doi: 10.1109/ICASSP55912.2026.11461248.
