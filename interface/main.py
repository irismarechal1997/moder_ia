## Import
import pandas as pd
from utils.registry import save_model, load_model
from utils.ml_baseline import baseline_model
from utils.dl import GRU_model, LSTM_model, Conv1D_model
from utils.bert_binary import  bert_model_1
from utils.classif_model import classif_cnn_model, classif_GRU_model, full_model_classif
from utils.data_bert_classif import bert_classif

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

def train_DL_model(model_name, processed=False): ### select Conv1D, or GRU, or LSTM


    if model_name == "Conv1D":
        if processed:
            model = Conv1D_model()
        else:
            model = Conv1D_model(processed = False)
        save_model(model, "Conv1D")
        print(f"✅ Model successfully saved locally")

    if model_name == "GRU":
        if processed:
            model = GRU_model()
        else:
            model = GRU_model(processed = False)
        save_model(model, "GRU")
        print(f"✅ Model successfully saved locally")

    if model_name == "LSTM":
        if processed:
            model = LSTM_model()
        else:
            model = LSTM_model(processed = False)
        save_model(model, "LSTM")
        print(f"✅ Model successfully saved locally")

    if model_name == "bert_binary":
        if processed:
            model = bert_model_1()
        else:
            model = bert_model_1(processed = False)
        save_model(model, "bert_binary_v1")
        print(f"✅ Model successfully saved locally")

    return model

### Predict with DeepL models

def pred_DL(X_pred: str = None, model_name=any) -> str:
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


### Classification models

def train_classif_model(model_name): ### select bert_classif, GRU_classif,CNN_classif, full_model_classif

    if model_name == "GRU_classif":
        model = classif_GRU_model()
        save_model(model, "GRU_classif")
        print(f"✅ Model successfully saved locally")

    if model_name == "CNN_classif":
        model = classif_cnn_model()
        save_model(model, "CNN_classif")
        print(f"✅ Model successfully saved locally")

    if model_name == "full_model_classif":
        model = full_model_classif()
        save_model(model, "full_model_classif")
        print(f"✅ Model successfully saved locally")


    if model_name == "bert_classif":
        model = bert_classif()
        save_model(model, "bert_classif")
        print(f"✅ Model successfully saved locally")

    return model

def pred_classif_model(X_pred: str = None, model_name=any) -> str:
    """
    Make a prediction using the latest trained model
    """
    if model_name == "GRU_classif":
        model = load_model("GRU_classif")

    if model_name == "bert_classif":
        model = load_model("bert_classif")

    if model_name == "CNN_classif":
        model = load_model("CNN_classif")

    if model_name == "full_model_classif":
        model = load_model("full_model_classif")


    X=[X_pred]
    y_pred = model.predict(X)

    if y_pred == 1:
        print("Your tweet is offensive")
    else:
        print("✅ Your tweet is not offensive")
    return y_pred


if __name__ == "__main__":
    # while True:
    #     X_pred = str(input("Enter a tweet: "))
    #     model_name = str(input("Enter model name between LSTM, GRU and Conv1D : "))
    #     pred_baseline(X_pred)
    #     train_DL_model(model_name, processed=False)
    #     pred_DL(model_name, X_pred)

    # model_name = str(input("Enter model name between LSTM, GRU and Conv1D, bert_binary : "))
    # train_DL_model(model_name,processed=False)

    model_name = str(input("Enter model name between bert_classif, GRU_classif,CNN_classif, full_model_classif : "))
    train_classif_model(model_name)
