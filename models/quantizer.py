import torch


def uniform(x, precision):
    x_max, x_min = x.max(), x.min()
    scale = (x_max - x_min) / (2 ** precision - 1)
    z = -(x_min / scale).round()

    x_int = (x / scale).round() + z
    x_Q = x_int.clamp(0, 2 ** precision - 1)

    x_float = (x_Q - z) * scale

    return x_float


def minteq(x, precisions, ratios):
    shape = x.shape
    x = x.flatten()
    x_ = x.clone()
    count, total = 0, x.shape[0]
    quantized_idxs = None
    for precision, ratio in zip(precisions, ratios):
        q = uniform(x_, precision)
        error = (q - x_).abs()
        if quantized_idxs is not None:
            error[quantized_idxs] = float('inf')

        k = round(total * ratio)
        idxs = error.topk(k, largest=False).indices
        x[idxs] = q[idxs]

        count += k
        quantized_idxs = torch.cat((quantized_idxs, idxs)) if quantized_idxs is not None else idxs
    assert count == total
    x = x.view(shape)

    return x


def posteq(x, precisions, ratios):
    pos, o = 0, x.shape[0]
    for precision, ratio in zip(precisions, ratios):
        width = round(o * ratio) // 2
        x[pos:pos+width] = uniform(x[pos:pos+width], precision)
        x[-(pos+width):-pos if pos > 0 else None] = uniform(x[-(pos+width):-pos if pos > 0 else None], precision)
        pos += width
    assert pos == o // 2

    return x
