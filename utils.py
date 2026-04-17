import pickle
import os

def save_model(tree, vocab, filename="trained_model.pkl"):
    """Lưu tree và vocab vào một file nhị phân."""
    model_data = {
        "tree": tree,
        "vocab": vocab
    }
    with open(filename, "wb") as file:  # "wb" là write binary
        pickle.dump(model_data, file)
    print(f"Đã lưu mô hình thành công vào file: {filename}")

def load_model(filename="trained_model.pkl"):
    """Tải tree và vocab từ file nhị phân lên."""
    if not os.path.exists(filename):
        return None, None
    
    with open(filename, "rb") as file:  # "rb" là read binary
        model_data = pickle.load(file)
    print(f"Đã tải mô hình từ file: {filename}")
    return model_data["tree"], model_data["vocab"]