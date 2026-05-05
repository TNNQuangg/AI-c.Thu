import math 
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import random

def class_counts(labels):
    unique, counts=np.unique(labels,return_counts=True)
    return dict(zip(unique,counts))

def entropy(labels):
    total=len(labels)
    if total==0:
        return 0.0
    
    _, counts=np.unique(labels,return_counts=True)
    probs=counts/total
    return -np.sum(probs*np.log2(probs))

def gini(labels):
    total=len(labels)
    if total==0:
        return 0.0
    
    _, counts = np.unique(labels,return_counts=True)
    probs=counts/total
    return 1.0 - np.sum(probs**2)

def split_dataset(X,y,feature_index,threshold=0.5):
    if sp.issparse(X):
        column_data=X[:, feature_index].toarray().flatten()
    else:
        column_data=X[:, feature_index]

    left_mask=column_data<= threshold
    right_mask= ~left_mask

    return X[left_mask], y[left_mask], X[right_mask],y[right_mask]

def best_split(x, y, criterion="entropy", max_features=None):
    n_samples, n_features = x.shape
    if n_samples == 0 or n_features == 0:
        return None, None, 0.0
    
    best_feature, best_threshold, best_gain = None, None, 0.0
    impurity_func = entropy if criterion == "entropy" else gini
    parent_impurity = impurity_func(y)
    
    # =========================================================
    # CHIẾN THUẬT CHE MẮT THÔNG MINH (TỰ ĐỘNG LẤY TỈ LỆ)
    # =========================================================
    # Bắt tất cả các chữ bắt đầu bằng "active_"
    if isinstance(max_features, str) and max_features.startswith("active_"):
        # Tự động cắt chuỗi lấy số (VD: "active_40" -> 40 -> 0.4)
        percent = int(max_features.split("_")[1]) / 100.0
        
        if sp.issparse(x):
            active_features = np.where(x.getnnz(axis=0) > 0)[0] 
        else:
            active_features = np.where(np.count_nonzero(x, axis=0) > 0)[0]
            
        if len(active_features) == 0:
            return None, None, 0.0 
            
        # Nhân với tỉ lệ phần trăm vừa lấy được
        n_subset = max(1, int(len(active_features) * percent))
        features_to_search = random.sample(list(active_features), n_subset)
        
    elif max_features == "sqrt":
        n_subset = max(1, int(math.sqrt(n_features)))
        features_to_search = random.sample(range(n_features), n_subset)
    elif max_features is None:
        features_to_search = range(n_features)
    else:
        features_to_search = range(n_features)
    # =========================================================
    
    for feature_index in tqdm(features_to_search, desc="Searching", leave=False):
        if sp.issparse(x):
            non_zero_vals = x[:, feature_index].data
        else:
            feature_vals = x[:, feature_index]
            non_zero_vals = feature_vals[feature_vals > 0]

        if len(non_zero_vals) == 0:
            thresholds = [0.0]
        else:
            thresholds = [
                0.0, 
                np.percentile(non_zero_vals, 25), 
                np.median(non_zero_vals), 
                np.percentile(non_zero_vals, 75)
            ]
            thresholds = list(set(thresholds))

        for threshold in thresholds:
            left_x, left_y, right_x, right_y = split_dataset(x, y, feature_index, threshold)
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_weight = len(left_y) / n_samples
            right_weight = len(right_y) / n_samples
            weighted_child_impurity = (left_weight * impurity_func(left_y) + right_weight * impurity_func(right_y))

            gain = parent_impurity - weighted_child_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
            
    return best_feature, best_threshold, best_gain
    
def majority_label(labels):
    counts=class_counts(labels)
    return max(counts, key=counts.get)

# THÊM THAM SỐ max_features VÀO HÀM DƯỚI DÂY:
def build_tree(x, y, depth=0, max_depth=20, min_samples_split=10, min_samples_leaf=10, criterion="entropy", max_features=None):
    if not sp.issparse(x):
        x = np.array(x)
    y = np.array(y)

    current_counts = class_counts(y)
    current_samples = len(y)

    unique_labels = np.unique(y)
    probabilities = {label: (current_counts.get(label, 0) / current_samples) for label in unique_labels}

    if len(np.unique(y)) == 1 or depth >= max_depth or current_samples < min_samples_split:
        return {"type": "leaf", "class": majority_label(y), "samples": current_samples, "counts": current_counts, "probabilities": probabilities}
    
    # TRUYỀN max_features XUỐNG best_split
    feature_index, threshold, gain = best_split(x, y, criterion, max_features)
    
    if feature_index is None or gain <= 1e-6:
        return {"type": "leaf", "class": majority_label(y), "samples": current_samples, "counts": current_counts, "probabilities": probabilities}
    
    left_x, left_y, right_x, right_y = split_dataset(x, y, feature_index, threshold)
    
    if len(left_y) < min_samples_leaf or len(right_y) < min_samples_leaf:
        return {"type": "leaf", "class": majority_label(y), "samples": current_samples, "counts": current_counts, "probabilities": probabilities}

    # TRUYỀN max_features VÀO CÁC LẦN GỌI ĐỆ QUY
    left_subtree = build_tree(left_x, left_y, depth + 1, max_depth, min_samples_split, min_samples_leaf, criterion, max_features)
    right_subtree = build_tree(right_x, right_y, depth + 1, max_depth, min_samples_split, min_samples_leaf, criterion, max_features)

    return {"type": "node", "feature_index": feature_index, "threshold": threshold, "gain": gain, "samples": current_samples, "counts": current_counts, "left": left_subtree, "right": right_subtree}

def predict_one(tree, x_row):
    if tree["type"] =="leaf":
        return tree["class"]
    
    val=x_row[0,tree["feature_index"]] if sp.issparse(x_row) else x_row[tree["feature_index"]]

    if val<=tree["threshold"]:
        return predict_one(tree["left"],x_row)
    else:
        return predict_one(tree["right"],x_row)
    
def predict(tree, X):
    if sp.issparse(X):
        X=X.tocsr()
    
    y_pred=[]
    for i in range(X.shape[0]):
        row=X[i]
        y_pred.append(predict_one(tree,row))
    return np.array(y_pred)

def predict_proba_one(tree, x_row, n_classes=3):
    """Đi xuống cây đến leaf, trả về vector xác suất [neg, neu, pos]
    dựa trên tỉ lệ thật sự của các mẫu trong leaf đó."""
    if tree["type"] == "leaf":
        probs = np.zeros(n_classes)
        total = tree["samples"]
        for label, count in tree["counts"].items():
            if int(label) < n_classes:
                probs[int(label)] = count / total
        return probs

    val = x_row[0, tree["feature_index"]] if sp.issparse(x_row) else x_row[tree["feature_index"]]
    if val <= tree["threshold"]:
        return predict_proba_one(tree["left"], x_row, n_classes)
    else:
        return predict_proba_one(tree["right"], x_row, n_classes)

def predict_proba(tree, X, n_classes=3):
    """Trả về ma trận (n_samples, n_classes) — xác suất của từng class cho mỗi mẫu."""
    if sp.issparse(X):
        X = X.tocsr()
    return np.array([
        predict_proba_one(tree, X[i], n_classes)
        for i in range(X.shape[0])
    ])

def label_to_text(label):
    mapping = {0: "negative",1: "neutral",2: "positive"}
    return mapping.get(label,"unknown")
