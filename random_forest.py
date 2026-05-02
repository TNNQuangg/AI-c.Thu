import numpy as np
import scipy.sparse as sp
from decision_tree import build_tree, predict

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
            max_features=max_features # Ép cây chỉ nhìn 1 phần từ vựng
        )
        forest.append(tree)
    return forest

def predict_random_forest(forest, X):
    """Lấy ý kiến biểu quyết từ tất cả các cây"""
    # 1. Thu thập dự đoán từ mọi cây. Kết quả là ma trận (n_trees, n_samples)
    tree_preds = np.array([predict(tree, X) for tree in forest])
    
    y_pred = []
    # 2. Bầu chọn theo số đông (Majority Voting) cho từng câu văn
    for i in range(X.shape[0]):
        counts = np.bincount(tree_preds[:, i])
        y_pred.append(np.argmax(counts))
        
    return np.array(y_pred)