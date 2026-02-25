# 🛡️ Toxic Comment Classifier

A **Streamlit web application** that classifies comments as toxic, threatening, obscene, insulting, or hateful using a **BILSTM model with GloVe embeddings**.

---

## Streamlit App Link -
https://toxiccommentclassification-lreyqdtwdtmuuzvqhbh8gl.streamlit.app/

---

## Features

- Detects **six toxicity types**:
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`
- Uses **GloVe word embeddings** for input representation.
- BILSTM model for sequential text classification.
- Minimal, user-friendly **Streamlit interface**.
- Real-time probability predictions and label decisions.
- Works locally or on **Streamlit Cloud**.
- NLTK-based text preprocessing (stopwords removal, lowercasing, punctuation cleanup).

---

## Project Structure

```text
Toxic_Comment_Classification/
├─ app.py
├─ Notebook/
│  ├─ model_training.ipynb
│  └─ preprocessing.ipynb
├─ Data/
│  ├─ train.csv
├─ models/
│  ├─ bilstm.keras
│  └─ thresholds.json
├─ processed/
│  ├─ tokenizer.pkl
│  ├─ embedding_matrix.npy
│  ├─ X.npy
│  └─ y.npy
├─ requirements.txt
└─ README.md

---

## Installation (Local)

1. **Clone the repository**:

```bash
git clone https://github.com/abhifg/Toxic_Comment_Classification.git
cd Toxic_Comment_Classification
```

2. **Create Virtual Environment**:

```bash
conda create toxic_comment_classifier
conda activate toxic_comment_classifier
```

3. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**

```bash
streamlit run app.py
```

---

## Project Details

**Architecture**: Bidirectional LSTM (BILSTM)

**Embedding**: Pre-trained GloVe embeddings (glove.6B.100d.txt)

**Max sequence length**: 200 tokens

**Preprocessing**:

Lowercasing

Removing URLs, numbers, special characters

Stopwords removal using NLTK

**Tokenizer**: Saved as tokenizer.pkl

**Thresholds**: Saved as thresholds.json for each label

