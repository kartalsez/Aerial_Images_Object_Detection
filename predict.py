import torch

import utils
import dataset
from model import FCNN
from utils import ClassLabel

def predict(model, data_loader, device, class_label):

    model.eval()

    # Tile accumulator
    y_full = torch.Tensor().cpu()

    for i, (x, y) in enumerate(data_loader):

        x = x.to(device=device)

        with torch.no_grad():

            y_pred = model(x)
            y_pred = y_pred.to(device=y_full.device)

            # Stack tiles along dim=0
            y_full = torch.cat((y_full, y_pred), dim=0)

    if class_label == ClassLabel.background:
        return torch.max(y_full, dim=1)[1]

    if class_label == ClassLabel.house:
        return torch.max(-y_full, dim=1)[1]

    #TODO: Subclass error
    raise ValueError('Unknown class label: {}'.format(class_label))
