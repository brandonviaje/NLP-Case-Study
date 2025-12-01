# Sentiment-Analysis

## **Comparative study of NLP neural network architectures in Sentiment Analysis.**

This project compares the performance of four neural network models: RNN, LSTM, GRU, and DistilBERT, on the task of binary sentiment classification. The goal is to evaluate their **accuracy, training time, and model complexity** on the same dataset and demonstrate the evolution of NLP architectures.

---

## Overview

This is a **comparative case study** of NLP models:

- **RNN:** Basic recurrent network for sequential text modeling.  
- **LSTM:** Handles long-term dependencies with memory gates.
- **GRU:** Gated recurrent unit for more efficient memory retention.  
- **DistilBERT:** Pretrained transformer, captures context globally using attention.

We train each model on the **IMDb Movie Reviews dataset** (25,000 training and 25,000 test reviews) and compare their **performance metrics**.

---

## Dataset

**IMDb Movie Reviews** (50,000 labeled reviews):

- 25,000 positive reviews (balanced positive/negative)  
- 25,000 negative reviews (balanced positive/negative)  
- Reviews are preprocessed via tokenization, padding, and vocabulary building for recurrent models, and via HuggingFace tokenizer for DistilBERT.

Source: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Models

| Model | Description |
|-------|-------------|
| RNN | Simple recurrent neural network for sequential text modeling. |
| LSTM | Long short-term memory; handles long-term dependencies. |
| GRU | Gated recurrent unit; faster and more efficient than LSTM. |
| DistilBERT | Transformer-based pretrained model; high accuracy and contextual understanding. |

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/ethan-warriner/Sentiment-Analysis-API.git
cd Sentiment-Analysis-API
```
---

### Tools & Libraries

<table>
  <tr>
    <td><a href="https://pytorch.org/" target="_blank"><img src="https://pytorch.org/assets/images/pytorch-logo.png" width="60" alt="PyTorch"/></a></td>
    <td><a href="https://huggingface.co/" target="_blank"><img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/huggingface-color.png" width="60" alt="HuggingFace"/></a></td>
    <td><a href="https://matplotlib.org/" target="_blank"><img src="https://matplotlib.org/3.1.1/_static/logo2_compressed.svg" width="60" alt="Matplotlib"/></a></td>
    <td><a href="https://www.python.org/" target="_blank"><img src="https://www.python.org/static/community_logos/python-logo.png" width="60" alt="Python"/></a></td>
  </tr>
</table>

---
The Jupyter Notebook can be found in the same repo:
<p align="center">
  <a href="Sentiment_Analysis.ipynb">
    <img src="https://img.shields.io/badge/Read-More-blue?style=for-the-badge&logo=readthedocs"/>
  </a>
</p>
