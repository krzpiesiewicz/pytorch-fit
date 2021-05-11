from .metrics import Metric
from .run import run


def evaluate(model, device, loss, data_loader, metrics: Metric = [],
             prefix="test_"):
    res, output = run(
        model,
        device,
        loss,
        data_loader,
        metrics,
        optimizer=None,
        batch_prefix="Testing: ",
        loss_metric_prefix=prefix,
    )
    return res
