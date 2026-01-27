# training done in kaggle

import pandas as pd
from fastai.text.all import *
torch.cuda.is_available()

print(f"Using device: {default_device()}")

def main():
    df = pd.read_csv("/kaggle/input/fake-news/fake_news.csv")

    dls = TextDataLoaders.from_df(
        df,
        text_col="text", # input
        label_col="label", # output
        valid_pct=0.2, # 20% for validation, 80% for training
        seed=42,
    )

    learn = text_classifier_learner(
        dls, # text->num
        AWD_LSTM, # pretrained language model
        metrics=[accuracy, Precision(), Recall()]
    )
    
    learn.fine_tune(1)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()

    learn.export("/kaggle/working/fake_news_model.pkl")

    print("Training complete")

if __name__ == "__main__":
    main()
