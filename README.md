

# Natural Language Understanding Projects (CSL7640)

This repository contains implementations of four major assignments focused on core Natural Language Understanding tasks including POS tagging, Named Entity Recognition, Machine Translation, and Sentiment Analysis using classical and deep learning techniques.

---

## 📌 Assignment 1: POS Tagging using HMMs

**Goal:** Implement Part-of-Speech tagging using various Hidden Markov Model (HMM) configurations and the Viterbi algorithm.

### Features:

* Trained on the Penn Treebank dataset.
* 36-tag and 4-tag models.
* First-order, second-order HMMs, and a variant using previous word context.
* Evaluation of tagging accuracy for each configuration.

📊 *Key Insight:* The 4-tag model achieved the best accuracy with First-order HMM + previous word context.

---

## 📌 Assignment 2: Named Entity Recognition (NER) using HMM

**Goal:** Perform NER using Bigram and Trigram HMMs on annotated Twitter data.

### Features:

* Entity tags: PER, LOC, ORG, MISC.
* Bigram and Trigram HMMs.
* Context-aware emission probabilities.
* Viterbi algorithm for sequence decoding.

📊 *Key Insight:* Bigram HMMs showed better performance overall, but Trigram models improved with contextual emission modeling.

---

## 📌 Assignment 3: English-to-Hindi Neural Machine Translation

**Goal:** Build and evaluate LSTM and Transformer-based NMT models.

### Features:

* Encoder-decoder architecture using LSTM and Transformer.
* Data preprocessing with tokenization, padding, embedding (FastText).
* BLEU score used for evaluation.
* Ablation studies and a proposed hybrid LSTM-Transformer model.

📊 *Key Insight:* Transformer outperformed LSTM in BLEU score and translation fluency, but a hybrid model combining both showed even better results.

---

## 📌 Assignment 4: Sentiment Classification using FFNN and LSTM

**Goal:** Perform binary and multiclass sentiment classification on IMDB, SemEval, and Twitter datasets.

### Features:

* Models: FFNN, LSTM.
* Tasks: Binary (IMDB), 3-class (SemEval, Twitter).
* Used 5-Fold Stratified Cross Validation for robustness.
* Preprocessing: tokenization, vocabulary filtering, sequence encoding.
* Plots and metrics for loss, accuracy, and F1-score.

📊 *Key Insight:* LSTM consistently outperformed FFNN, especially for sequential and nuanced sentiment data like Twitter.

---

## 👨‍💻 Team Members

* Pooja Naveen Poonia (M24CSA020)
* Pranjal Malik (M24CSA021)
* Shivani Tiwari (M24CSA029)
* Suvigya Sharma (M24CSA033)



