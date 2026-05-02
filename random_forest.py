import numpy as np
import scipy.sparse as sp
from decision_tree import build_tree, predict, predict_proba

def bootstrap_sample(X, y):
    """Lấy ngẫu nhiên các mẫu (có hoàn lại) để tạo tập huấn luyện mới cho 1 cây"""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

def build_random_forest(X, y, n_trees=5, max_depth=20, min_samples_leaf=10, max_features="sqrt"):
    """Trồng rừng ngẫu nhiên"""
    forest = []
    for i in range(n_trees):
        print(f"\n🌳 Đang trồng cây thứ {i+1}/{n_trees}...")
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = build_tree(
            X_sample, y_sample,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )
        forest.append(tree)
    return forest

def predict_random_forest(forest, X, n_classes=3, neutral_threshold=0.5):
    """
    Soft voting: cộng xác suất thật sự từ leaf của mỗi cây, chia trung bình.

    Logic neutral threshold:
        - Nếu class tự tin nhất là Neutral (1)
          VÀ xác suất Neutral trung bình < neutral_threshold
          → chuyển sang Neg(0) hoặc Pos(2), tùy cái nào cao hơn

    Trả về:
        y_pred    : mảng nhãn dự đoán (n_samples,)
        all_probs : ma trận xác suất trung bình (n_samples, n_classes)
                    — dùng để debug hoặc hiển thị độ tự tin
    """
    # Cộng xác suất từ tất cả cây → (n_samples, n_classes)
    all_probs = np.zeros((X.shape[0], n_classes))
    for tree in forest:
        all_probs += predict_proba(tree, X, n_classes)

    # Chuẩn hóa về xác suất trung bình
    all_probs /= len(forest)

    y_pred = []
    for probs in all_probs:
        predicted = np.argmax(probs)

        # Nếu predict Neutral nhưng không đủ tự tin → chuyển sang Neg/Pos
        if predicted == 1 and probs[1] < neutral_threshold:
            predicted = 0 if probs[0] >= probs[2] else 2

        y_pred.append(predicted)

    return np.array(y_pred), all_probs