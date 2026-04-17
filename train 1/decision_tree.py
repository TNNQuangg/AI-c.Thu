import math 
import numpy as np
from tqdm import tqdm

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
    left_mask=X[:, feature_index]<= threshold
    right_mask= ~left_mask

    return X[left_mask], y[left_mask], X[right_mask],y[right_mask]

def best_split(x,y, criterion="entropy"):
    n_samples, n_features=x.shape
    if n_samples ==0 or n_features==0:
        return None,None,0.0
    
    best_feature,best_threshold,best_gain=None,None,0.0
    impurity_func = entropy if criterion =="entropy" else gini
    parent_impurity=impurity_func(y)
    
    for feature_index in tqdm(range(n_features),desc="Searching",leave=False):
        feature_vals=x[:,feature_index]
        non_zero_vals=feature_vals[feature_vals>0]

        if len(non_zero_vals)==0:
            thresholds=[0.0]
        else:
            thresholds=[0.0,np.mean(non_zero_vals)]

        for threshold in thresholds:
            left_x,left_y,right_x,right_y=split_dataset(x,y,feature_index,threshold)
            if len(left_y)==0 or len(right_y)==0:
                continue

            left_weight=len(left_y)/n_samples
            right_weight=len(right_y)/n_samples
            weighted_child_impurity=(left_weight*impurity_func(left_y)+right_weight*impurity_func(right_y))

            gain=parent_impurity-weighted_child_impurity

            if gain>best_gain:
                best_gain=gain
                best_feature=feature_index
                best_threshold=threshold
            
    return best_feature, best_threshold,best_gain
    
def majority_label(labels):
    counts=class_counts(labels)
    return max(counts, key=counts.get)

def build_tree(x,y,depth=0, max_depth=12,min_samples_split=2,min_samples_leaf=5,criterion="entropy"):
    
    x=np.array(x)
    y=np.array(y)

    current_counts=class_counts(y)
    current_samples=len(y)

    unique_labels=np.unique(y)
    probabilities={label: (current_counts.get(label,0)/current_samples) for label in unique_labels}

    if len(np.unique(y))==1 or depth>=max_depth or current_samples < min_samples_split:
        return {
            "type": "leaf",
            "class": majority_label(y),
            "samples":current_samples,
            "counts": current_counts,
            "probabilities":probabilities
        }
    
    feature_index,threshold,gain=best_split(x,y,criterion)
    
    if feature_index is None or gain<=1e-6:
        return {
            "type": "leaf",
            "class": majority_label(y),
            "samples":current_samples,
            "counts": current_counts,
            "probabilities":probabilities
        }
    
    left_x, left_y, right_x, right_y = split_dataset(x,y, feature_index,threshold)
    
    if len(left_y)<min_samples_leaf or len(right_y)<min_samples_leaf:
        return{
            "type":"leaf",
            "class":majority_label(y),
            "samples": current_samples,
            "counts": current_counts,
            "probabilities":probabilities
        }

    left_subtree= build_tree(
        left_x,
        left_y,
        depth+1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        criterion
    )

    right_subtree= build_tree(
        right_x,
        right_y,
        depth+1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        criterion
    )

    return {
        "type": "node",
        "feature_index": feature_index,
        "threshold":threshold,
        "gain": gain,
        "samples":current_samples,
        "counts": current_counts,
        "left": left_subtree,
        "right": right_subtree
    }

def predict_one(tree, x):
    if tree["type"] =="leaf":
        return tree["class"]

    if x[tree["feature_index"]]<=tree["threshold"]:
        return predict_one(tree["left"],x)
    else:
        return predict_one(tree["right"],x)
    
def predict(tree, X):
    X=np.array(X)
    return np.array([predict_one(tree,x) for x in X])
    
def label_to_text(label):
    mapping = {0: "negative",1: "neutral",2: "positive"}
    return mapping.get(label,"unknown")

def print_tree(tree, vocab_reverse, depth=0, branch_name="root"):
    indent = "    " * depth
    if tree["type"] == "leaf":
        probs = tree.get("probabilities", {0: 0.0, 1: 0.0})
        print(f"{indent}[{branch_name}] Leaf -> Predict: {tree['class']} ({label_to_text(tree['class'])}), Samples: {tree['samples']}, P(pos)={probs.get(1, 0.0):.2f}")
        return

    feat_word = vocab_reverse.get(tree["feature_index"], f"feature_{tree['feature_index']}")
    print(f"{indent}[{branch_name}] Node -> check '{feat_word}' <= {tree['threshold']} (gain: {tree['gain']:.4f}, samples: {tree['samples']})")
    print(f"{indent}    if '{feat_word}' <= {tree['threshold']}:")
    print_tree(tree["left"], vocab_reverse, depth + 1, branch_name="left")
    print(f"{indent}    else:")
    print_tree(tree["right"], vocab_reverse, depth + 1, branch_name="right")