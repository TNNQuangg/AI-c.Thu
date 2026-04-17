import numpy as np

def accuracy_score(y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    if len(y_true) ==0:
        return 0.0
    return np.mean(y_true==y_pred)

def precision_score(y_true,y_pred,):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    classes=np.unique(y_true)
    precisions=[]

    for cls in classes:
        tp=np.sum((y_true==cls)&(y_pred==cls))
        fp=np.sum((y_true!=cls)&(y_pred==cls))
        if tp+fp>0:
            precisions.append(tp/(tp+fp))
        else:
            precisions.append(0.0)
    
    return np.mean(precisions) if precisions else 0.0

def recall_score (y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    classes=np.unique(y_true)
    recalls=[]

    for cls in classes:
        tp=np.sum((y_true==cls)&(y_pred==cls))
        fn=np.sum((y_true==cls)&(y_pred!=cls))
        if tp+fn>0:
            recalls.append(tp/(tp+fn))
        else:
            recalls.append(0.0)
    return np.mean(recalls) if recalls else 0.0

def f1_score(y_true,y_pred):
    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)

    if (precision+recall)==0:
        return 0.0
    
    return 2*precision*recall/(precision+recall)

def confusion_matrix_multi(y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    classes=np.unique(np.concatenate([y_true,y_pred]))
    n_classes=len(classes)

    matrix=np.zeros((n_classes,n_classes),dtype=int)

    class_to_idx={cls:i for i,cls in enumerate(classes)}

    for t,p in zip(y_true,y_pred):
        matrix[class_to_idx[t],class_to_idx[p]]+=1
    
    return matrix,classes