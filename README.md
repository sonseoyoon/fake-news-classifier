# Fake vs Real News Classifier

This project implements a **fake news detection system** using a pretrained language model fine-tuned with **fastai**.  
Given the text of a news article, the model predicts whether it is **FAKE** or **REAL**, along with a confidence score.

The goal of this project is to explore how **natural language processing (NLP)** can be used to identify fake news based on **linguistic patterns** such as writing style, tone, and structure.  
Note: The model does **not** fact-check content; it classifies articles based on how they are written, not whether the claims are objectively true.

A **Streamlit web application** is provided so users can interactively test the model by pasting article text.

## Dataset
The dataset consists of records, each containing text of a news article and one of the following labels:
- `FAKE`
- `REAL`

---

## Model & Approach
- **Framework:** fastai (PyTorch)
- **Model:** AWD-LSTM (pretrained language model)
- **Task:** Binary text classification

---

## Training & Deployment
- Training was performed on **Kaggle** using **GPU acceleration**.
- The trained model was exported as a `.pkl` file.
- The model is deployed locally and online using **Streamlit**.

---

## Running the App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
fake-news-classifier/
├── data/
│   └── fake_news.csv
├── models/
│   └── fake_news_model.pkl
├── prepare_data.py
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
