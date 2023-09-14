
import pickle


def save_model(model, filename):
    # Save model locally
    pickle.dump(model, open(filename, 'wb'))
    print("✅ Model saved locally")

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


# Score_LSTM=85.167%
# Score_GRU=85.59%
# Score_Conv1D=62.6%
