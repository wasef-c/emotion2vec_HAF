import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def calculate_uar(cm):
    """Calculate Unweighted Average Recall (UAR) from confusion matrix."""
    recalls = cm.diagonal() / cm.sum(axis=1)
    return np.mean(recalls)


def calculate_wa(cm):
    """
    Calculate Weighted Accuracy (WA) from confusion matrix.
    WA = sum over classes of (support_i * recall_i) / total_samples
    """
    # support (number of true samples per class) is the sum over rows
    support = cm.sum(axis=1)
    # recall per class: TP / (TP + FN)
    recall_per_class = np.diag(cm) / support
    # weighted sum of recalls
    weighted_accuracy = np.sum(recall_per_class * support) / np.sum(cm)
    return weighted_accuracy


def calculate_comprehensive_metrics(all_labels, all_preds, class_names=None):
    """
    Calculate all emotion recognition metrics: Accuracy, UAR, WA, F1
    """
    if class_names is None:
        class_names = ["neutral", "happy", "sad", "anger"]

    # Basic metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)

    # Advanced metrics
    uar = calculate_uar(cm)
    # This should equal accuracy, but good for verification
    wa = calculate_wa(cm)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    # Class-wise accuracies (recalls)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    return {
        "accuracy": accuracy,
        "wa": wa,  # Weighted Accuracy
        "uar": uar,  # Unweighted Average Recall
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_accuracies": {
            class_names[i]: float(acc) for i, acc in enumerate(class_accuracies)
        },
    }