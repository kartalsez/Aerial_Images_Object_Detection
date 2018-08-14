import torch
import torch.optim

import utils
import dataset
from model import FCNN
from loss import CrossEntropyLoss2d

def train(model, train_loader, device,
          tile_size, epochs=10, batch_size=1,
          learning_rate=1e-4, momentum=0.9, weight_decay=5e-3):

    criterion = CrossEntropyLoss2d()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    model.train()

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    training_stats = utils.Stats()

    for n in range(epochs):

        epoch_stats = utils.Stats()

        loader_with_progress = utils.loader_with_progress(train_loader,
            epoch_n=n, epoch_total=epochs, stats=epoch_stats, leave=True)

        for i, (x, y) in enumerate(loader_with_progress):

            y = y.to(device=device)
            x = x.to(device=device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            epoch_stats.append_loss(loss.item())
            training_stats.append_loss(loss.item())

            loader_with_progress.set_postfix(epoch_stats.fmt_dict())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, training_stats
