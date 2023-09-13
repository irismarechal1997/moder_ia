
## Import

from utils.ml_baseline import baseline_model
import pandas as pd

## Baseline_model

data_processed = pd.read_csv("data/processed_dataset_v1.csv") # assign variable

# train_baseline_model
def train_baseline_model(processed=True):

    if processed:
        model = baseline_model()

    else:
        model = baseline_model(processed=False)

    return model



X_pred = str(input("Enter a tweet: "))

def pred_baseline(X_pred: str = None) -> str:
    """
    Make a prediction using the latest trained model
    """
    model = baseline_model(processed = False)

    print("\n⭐️ Use case: predict")

    X=[X_pred]
    y_pred = model.predict(X)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    if y_pred == 1:
        print("Your tweet is offenseive")
    return y_pred


if __name__ == "__main__": ## dire quelle fonction
        pred_baseline(X_pred)
