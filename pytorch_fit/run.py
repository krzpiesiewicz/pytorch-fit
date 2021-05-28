import time

import torch

from .metrics import Metric


def print_progress(batch_idx, batch_size, total_size, prefix="", suffix=""):
    current = batch_idx * batch_size
    ratio = current / total_size
    shift = 0
    n = 10
    while n <= total_size:
        shift += current < n
        n *= 10
    bar_numbers = "%d/%d(%d%%)" % (current, total_size, int(100 * ratio))
    max_numbers_len = len("%d/%d(%d%%)" % (total_size, total_size, 100))
    total = 100 - len(suffix) - len(prefix)
    numbers_start_pos = int((total - max_numbers_len) / 2) + shift
    numbers_end_pos = numbers_start_pos + len(bar_numbers)
    progress = int(ratio * total)
    if progress == total:
        progress_bar = "=" * total
    else:
        progress_bar = "=" * (progress - 1) + ">" + " " * (total - progress)
    progress_bar = (
            progress_bar[:numbers_start_pos] + bar_numbers +
            progress_bar[numbers_end_pos:]
    )
    print("\r" + "%s[%s] %s" % (prefix, progress_bar, suffix), end="")


def run(model, device, loss, data_loader, metrics: Metric = [], optimizer=None,
        title=None, batch_prefix="", loss_metric_prefix="", start_time=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    losses = []

    for metric in metrics:
        metric.reset_state()

    total_size = len(data_loader.dataset)
    if title is not None:
        print(title)
    print_progress(0, 0, total_size, batch_prefix)

    with torch.autograd.set_grad_enabled(optimizer is not None):
        for batch_idx, (data, target) in enumerate(data_loader):
            batch_size = len(data)
            data, target = data.to(device), target.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            output = model(data)
            loss_value = loss(output, target)
            if optimizer is not None:
                loss_value.backward()
                optimizer.step()
            losses.append(loss_value.item())
            loss_value = sum(losses) / (len(losses) * batch_size)
            suffix = f" loss: {loss_value:.3f}"

            for metric in metrics:
                metric.update_state(output, target)
                if metric.is_value_simple():
                    suffix += f", {metric.short_name()}: {metric.value():.3f}"
            if start_time is not None:
                suffix += f", elapsed time: {time.time() - start_time:.1f}s"
            print_progress(batch_idx + 1, batch_size, total_size,
                           batch_prefix, suffix)
    res = {"Loss": {f"{loss_metric_prefix}loss": [loss_value]}}
    output = f" {loss_metric_prefix}loss: {loss_value:.3f}"
    for metric in metrics:
        metric_value = metric.value()
        res[metric.name()] = {
            f"{loss_metric_prefix}{metric.short_name()}": [metric_value]
        }
        if metric.is_value_simple():
            output += f", {loss_metric_prefix}{metric.short_name()}: {metric_value:.3f}"

    return res, output
