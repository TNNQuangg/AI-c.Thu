import csv
import pandas as pd
import numpy as np
import random
from utils import save_model, load_model
from vectorizer import build_vocabulary, texts_to_matrix, text_to_vector
from decision_tree import build_tree, predict, print_tree
from metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix_multi
)

def train_test_split_texts(texts, labels, test_size=0.2, seed=42):
    # Dùng numpy để trộn và chia tách mảng tiện lợi hơn
    np.random.seed(seed)
    indices = np.random.permutation(len(texts))
    split_idx = int(len(texts) * (1 - test_size))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return (
        [texts[i] for i in train_idx], [texts[i] for i in test_idx],
        labels[train_idx], labels[test_idx]
    )

def main():
    MODEL_FILE = "trained_model.pkl"
    DATASET_FILE = "dataset_fin.csv" # Đặt biến dùng chung để tránh nhầm lẫn
    
    # Thử tải mô hình đã lưu
    tree, vocab = load_model(MODEL_FILE)

    if tree is None or vocab is None:
        print("Chưa có mô hình. Bắt đầu quá trình huấn luyện từ đầu...")
        
        # 1. Load dữ liệu và chia tập train/test
        df = pd.read_csv(DATASET_FILE, encoding="latin-1")
        # 1. Xóa bỏ các dòng bị thiếu dữ liệu ở cột 'text' hoặc 'label'
        df = df.dropna(subset=["text", "label"])
        
        # 2. Ép kiểu toàn bộ cột text về dạng chuỗi (string)
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)

        texts = df["text"].tolist()
        labels = df["label"].values
        
        train_texts, test_texts, y_train, y_test = train_test_split_texts(
            texts, labels, test_size=0.2, seed=42
        )

        # 2. Xây dựng vocab và vector hóa
        vocab = build_vocabulary(train_texts, min_freq=8, remove_stopwords=True)
        X_train = np.array(texts_to_matrix(train_texts, vocab, remove_stopwords=True))
        
        # 3. Train mô hình
        print(f"Đang huấn luyện... (Kích thước tập train: {X_train.shape})")
        tree = build_tree(X_train, y_train, max_depth=25, min_samples_split=10, min_samples_leaf=10, criterion="entropy")
        
        # 4. LƯU MÔ HÌNH LẠI ĐỂ LẦN SAU DÙNG
        save_model(tree, vocab, MODEL_FILE)
        
    else:
        # Nếu đã có mô hình, ta vẫn cần load test data để in ra file metrics đánh giá
        print("Bỏ qua bước huấn luyện, sử dụng mô hình đã lưu.")
        df = pd.read_csv(DATASET_FILE, encoding="latin-1") # Đồng bộ file data
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        texts = df["text"].tolist()
        labels = df["label"].values
        _, test_texts, _, y_test = train_test_split_texts(
            texts, labels, test_size=0.2, seed=42
        )

    # Dù là model mới train hay model load lên, ta đều có thể test
    print("\n===== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST =====")
    X_test = np.array(texts_to_matrix(test_texts, vocab, remove_stopwords=True))
    y_pred = predict(tree, X_test)
    
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    matrix, classes = confusion_matrix_multi(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("Labels:", classes)
    print(matrix)
    
    # Test với một câu mới hoàn toàn
    print("\n===== TEST THỰC TẾ =====")
    sample_text = "i love this book"
    sample_vector = np.array([text_to_vector(sample_text, vocab, remove_stopwords=True)])
    prediction = predict(tree, sample_vector)[0]
    
    # Cập nhật ánh xạ nhãn cho 3 lớp
    label_map = {0: "Negative (0)", 1: "Neutral (1)", 2: "Positive (2)"}
    
    print(f"Câu: '{sample_text}'")
    print(f"Dự đoán: {label_map.get(prediction, 'Unknown')}")

if __name__ == "__main__":
    main()