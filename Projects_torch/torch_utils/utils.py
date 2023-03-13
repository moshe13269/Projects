
import torch
from torchviz import make_dot


def save_plot_model(model, input_shape=(1, 130, 129), path2save=None):
    x = torch.randn(input_shape).requires_grad_(True)
    mask = torch.zeros((65, 65)).requires_grad_(True)
    y = model([x, mask])
    make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)])).render("attached", format="png")

    # show_attrs = True,
    # show_saved = True
    """
    https://github.com/szagoruyko/pytorchviz
    """



