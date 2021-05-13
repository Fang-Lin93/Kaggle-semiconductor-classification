
import matplotlib.pyplot as plt
from sklearn import metrics


def tensor2img(tensor, defect_area=None):
    """
    torch tensor of shape (C, H, W) = (1, 267, 275)
    defect_area: e.g. = (87, 137, 118, 174)
    """
    plt.imshow(tensor.squeeze(0), cmap='gray')
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tensor.squeeze(0), cmap='gray')
    if defect_area:
        x1, y1, x2, y2 = defect_area
        anchor = (x1, y1)
        h = y2 - y1
        w = x2 - x1
        ax.add_patch(plt.Rectangle(anchor, w, h, edgecolor='red', facecolor='none'))

    fig.show()
    return fig


def plot_roc(pred_y, y):
    """
    AUC ROC
    """
    fpr, tpr, threshold = metrics.roc_curve(y, pred_y)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.6f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc



