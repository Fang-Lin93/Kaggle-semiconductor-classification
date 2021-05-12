
import matplotlib.pyplot as plt


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



