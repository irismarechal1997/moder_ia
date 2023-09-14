
## Import

from utils.ml_baseline import baseline_model
import pandas as pd
from utils.registry import save_model, load_model
from utils.dl import GRU_model, LSTM_model, Conv1D_modelGo

### Baseline_model

data_processed = pd.read_csv("data/processed_dataset_v1.csv") # assign variable
# train_baseline_model

def train_baseline_model(processed=False):

    if processed:
        model = baseline_model()

    else:
        model = baseline_model(processed=False)

    save_model(model, "ml_model")
    return model

# predict_baseline_model
def pred_baseline(X_pred: str = None) -> str:
    """
    Make a prediction using the latest trained model
    """

    model = load_model("ml_model")
    X=[X_pred]
    y_pred = model.predict(X)

    if y_pred == 1:
        print("Your tweet is offensive")
    else:
        print("✅ Your tweet is not offensive")
    return y_pred


### Train DeepL models

def train_DL_model(processed=False, model_name): ### select Conv1D, or GRU, or LSTM

    if model_name == "Conv1D":
        if processed:
            model = Conv1D_modelGo()
        else:
            model = Conv1D_modelGo(processed = False)
        save_model(model, "Conv1D")
        print(f"✅ Model {model} successfully saved locally")

    if model_name == "GRU":
        if processed:
            model = GRU_model()
        else:
            model = GRU_model(processed = False)
        save_model(model, "GRU")
        print(f"✅ Model {model} successfully saved locally")

    if model_name == "LSTM":
        if processed:
            model = LSTM_model()
        else:
            model = LSTM_model(processed = False)
        save_model(model, "LSTM")
        print(f"✅ Model {model} successfully saved locally")

    return model

### Predict with  DeepL models

def pred_DL(X_pred: str = None, model_name) -> str:
    """
    Make a prediction using the latest trained model
    """
    if model_name == "Conv1D":
        model = load_model("Conv1D")

    if model_name == "GRU":
        model = load_model("GRU")

    if model_name == "LSTM":
        model = load_model("LSTM")

    X=[X_pred]
    y_pred = model.predict(X)

    if y_pred == 1:
        print("Your tweet is offensive")
    else:
        print("✅ Your tweet is not offensive")
    return y_pred


if __name__ == "__main__":
    while True:
        X_pred = str(input("Enter a tweet: "))
        pred_baseline(X_pred)
        train_DL_model(processed=False, model_name)
        pred_DL(X_pred: str = None, model_name)
