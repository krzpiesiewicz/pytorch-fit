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


def run(
        model,
        device,
        loss,
        data_loader,
        metrics: Metric = [],
        optimizer=None,
        title=None,
        batch_prefix="",
        loss_metric_prefix="",
        precision=3,
        precision_big=0,
        reduction="mean",
        start_time=None,
        return_batches_outputs=False,
        reset=False,
        reset_after_train_batch=False,
        callback_before_train_batch=None,
        callback_after_train_batch=None,
        mute=False,
        debug=False
):
    assert reduction in ["mean", "sum"]
    if optimizer is None:
        model.eval()
    else:
        model.train()

    if reset:
        model.reset()

    losses = []

    for metric in metrics:
        metric.reset_state()

    total_size = len(data_loader.dataset)
    if title is not None:
        print(title)
    print_progress(0, 0, total_size, batch_prefix)

    batches_outputs = []

    def value_precision_str(value):
        value_precision = precision_big if value >= 100 else precision
        if value_precision == 0:
            return str(int(value))
        else:
            return f"{value:.{value_precision}f}"

    with torch.autograd.set_grad_enabled(optimizer is not None):
        for batch_idx, (data, target) in enumerate(data_loader):
            if type(data) is not list:
                data = [data]
            batch_size = len(data[0])
            if debug:
                print(f"batch_size: {batch_size}")
            data = [tensor.to(device) for tensor in data]
            if type(target) is list:
                metric_details_list = target[1:]
                target = target[0]
            else:
                metric_details_list = []
            target = target.to(device)

            if optimizer is not None:
                if callback_before_train_batch is not None:
                    callback_before_train_batch()
                optimizer.zero_grad()
            batch_output = model(*data)
            loss_value = loss(batch_output, target)
            if optimizer is not None:
                loss_value.backward()
                optimizer.step()
                if reset_after_train_batch:
                    model.reset()
                if callback_after_train_batch is not None:
                    callback_after_train_batch()
            losses.append(loss_value.item())
            loss_value = sum(losses) / len(losses)  # reduction == "sum"
            if reduction == "mean":
                loss_value = loss_value / batch_size
            loss_value_str = value_precision_str(loss_value)
            suffix = f" loss: {loss_value_str}"

            for metric in metrics:
                metric.update_state(batch_output, target, *metric_details_list)
                if metric.is_value_simple():
                    metric_value_str = value_precision_str(metric.value())
                    suffix += f", {metric.short_name()}: {metric_value_str}"
            if start_time is not None:
                suffix += f", elapsed time: {time.time() - start_time:.1f}s"
            if not mute:
                print_progress(batch_idx + 1, batch_size, total_size,
                               batch_prefix, suffix)
            if return_batches_outputs:
                batches_outputs.append(batch_output)

    res = {"Loss": {f"{loss_metric_prefix}loss": [loss_value]}}
    loss_value_str = value_precision_str(loss_value)
    stdin_output = f" {loss_metric_prefix}loss: {loss_value_str}"
    for metric in metrics:
        metric_value = metric.value()
        res[metric.name()] = {
            f"{loss_metric_prefix}{metric.short_name()}": [metric_value]
        }
        if metric.is_value_simple():
            metric_value_str = value_precision_str(metric_value)
            stdin_output += f", {loss_metric_prefix}{metric.short_name()}: " \
                            f"{metric_value_str}"

    if return_batches_outputs:
        return res, stdin_output, batches_outputs
    else:
        return res, stdin_output
