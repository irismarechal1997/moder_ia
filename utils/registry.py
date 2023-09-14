
import pickle


def save_model(model, filename):
    # Save model locally
    pickle.dump(model, open(filename, 'wb'))
    print("✅ Model saved locally")

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
