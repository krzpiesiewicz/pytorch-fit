import time

from torch import optim

from .history import add_to_history
from .run import run


def train_epoch(model, device, optimizer, loss, train_loader, val_loader=None,
                metrics=[], metrics_val_only=False, epoch=1, history=None,
                newline=True, start_time=None, **kwargs):
    train_metrics = [] if metrics_val_only else metrics
    if start_time is None:
        start_time = time.time()
    res, train_output = run(
        model,
        device,
        loss,
        train_loader,
        train_metrics,
        optimizer,
        batch_prefix=f"\rEpoch {epoch:4}:   Training: ",
        loss_metric_prefix="train_",
        start_time=start_time,
        **kwargs
    )
    if val_loader is not None:
        val_res, val_output = run(
            model,
            device,
            loss,
            val_loader,
            metrics,
            start_time=start_time,
            batch_prefix=f"\rEpoch {epoch:4}: Validating: ",
            loss_metric_prefix="val_",
            **kwargs
        )
        add_to_history(res, val_res)
    print(
        f"\rEpoch {epoch:4}:  {train_output},{val_output}, elapsed time: "
        f"{time.time() - start_time:.1f}s  ",
        end="\n" if newline else "",
    )
    if history is None:
        return res
    else:
        add_to_history(history, res)


def fit(network, device, optimizer, loss, train_loader, val_loader=None,
        metrics=[], lr=None, n_epochs=300, initial_epoch=1, last_epoch=None,
        history=None, stop_cond=None, **kwargs):
    if history is None:
        history = {}
    if n_epochs > 50:
        interval = 10
    else:
        interval = 1

    if type(optimizer) is str:
        if optimizer == "Adam":
            optimizer = optim.Adam(network.parameters())
        if optimizer == "SGD":
            optimizer = optim.SGD(network.parameters(), lr=1e-3)
    if lr is not None:
        for g in optimizer.param_groups:
            g["lr"] = 1e-3

    if last_epoch is not None:
        initial_epoch = last_epoch + 1
        n_epochs -= last_epoch

    for epoch in range(initial_epoch, initial_epoch + n_epochs):
        if interval > 1:
            if epoch % interval == 1 or epoch == initial_epoch:
                start_time = time.time()
            show = epoch % interval == 0
        else:
            start_time = time.time()
            show = True
        train_epoch(network, device, optimizer, loss, train_loader,
                    val_loader, metrics, epoch=epoch, history=history,
                    newline=show, start_time=start_time, **kwargs)
        if stop_cond and stop_cond(history):
            break
    return history
