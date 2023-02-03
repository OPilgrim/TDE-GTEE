import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, jaccard_score, precision_recall_fscore_support


def draw_p_r_curve(recall, precision, img_path):
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.savefig('./checkpoints/imgs/{}.png'.format(img_path))


def evaluate(y_pred, labels, preds=None, img_path=None):
    TP, TN, FP, FN = 0, 0, 0, 0
    for p, l in zip(y_pred, labels):
        TP += (p == 1) & (l == 1)
        TN += (p == 0) & (l == 0)
        FN += (p == 0) & (l == 1)
        FP += (p == 1) & (l == 0)
    
    p = TP / (TP + FP) if (TP + FP) else 0
    r = TP / (TP + FN) if (TP + FN) else 0

    print("="*50)
    ALL = TP + FP + TN + FN
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP/ALL, TN/ALL, FP/ALL, FN/ALL))
    print("="*50)

    acc = accuracy_score(y_pred, labels)
    f1 = f1_score(y_pred, labels)
    
    if preds is not None:
       precision, recall, thresholds = precision_recall_curve(labels, preds)
       draw_p_r_curve(recall, precision, img_path)

    return acc, p, r, f1


def multi2binary(y_true, y_score):
    b_labels = list()
    b_preds = list()
    for x, y in zip (y_score, y_true):
        b_labels.append(0 if y[-1] else 1)
        b_preds.append(0 if np.argmax(x) == len(x)-1 else 1)
        
    return evaluate(b_preds, b_labels)


def multilabel_evaluate(y_true, y_pred, y_score, mode=None, img_path=None):
    mic_p, mic_r, mic_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
                '{}_mic_f1'.format(mode) : mic_f1,
                '{}_mic_p'.format(mode) : mic_p,
                '{}_mic_r'.format(mode) : mic_r,
                '{}_mac_f1'.format(mode): mac_f1,
                '{}_mac_p'.format(mode): mac_p,
                '{}_mac_r'.format(mode): mac_r,
    }

    return metrics