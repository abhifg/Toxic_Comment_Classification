# 🛡️ Toxic Comment Classifier

A **Streamlit web application** that classifies comments as toxic, threatening, obscene, insulting, or hateful using a **BILSTM model with GloVe embeddings**.

---

## Streamlit App Link -
https://toxiccommentclassification-lreyqdtwdtmuuzvqhbh8gl.streamlit.app/

---

## Dataset
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- 159,571 Wikipedia talk page comments
- 6 toxicity labels
  
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
```
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

## Usage

1. Open the app in your browser.

2. Enter a comment in the text area.

3. Click Analyze.

4. Results:

✅ Comment appears safe

⚠️ Toxic content detected

Detailed results show probability for each label and whether the model predicts it as toxic.

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

---

## Model Performance
| Metric | Score |
|--------|-------|
| Validation AUC | 0.9837 |
| Validation Accuracy | 99.27% |
| Best Epoch | 4 |

### Per Label F1 Score (with threshold tuning)
| Label | F1 Score |
|-------|----------|
| Toxic | 0.77 |
| Severe Toxic | 0.48 |
| Obscene | 0.79 |
| Threat | 0.41 |
| Insult | 0.68 |
| Identity Hate | 0.56 |

---

## License

MIT License – free to use, modify, and distribute.
