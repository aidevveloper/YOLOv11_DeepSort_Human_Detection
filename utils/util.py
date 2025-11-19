import copy
import random
from time import time

import math
import numpy
import cv2
import torch
import torchvision
from torch.nn.functional import cross_entropy

from scipy import linalg
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------
# YOLOv11: Util 1
# -----------------------------------------------------------------

chi2inv95 = {1: 3.8415,
             2: 5.9915,
             3: 7.8147,
             4: 9.4877,
             5: 11.070,
             6: 12.592,
             7: 14.067,
             8: 15.507,
             9: 16.919}

INFTY_COST = 1e+5


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float()
    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      f='./weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')
    # Inference example
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py


def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xy2wh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, shape2[1])  # x1
    coords[:, 1].clamp_(0, shape2[0])  # y1
    coords[:, 2].clamp_(0, shape2[1])  # x2
    coords[:, 3].clamp_(0, shape2[0])  # y2
    return coords


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.65):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx6 (box, confidence, cls)
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes, scores
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        indices = indices[:max_det]  # limit detections

        output[index] = x[indices]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output


def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, names, save_dir):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = numpy.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    pyplot.close(fig)


def plot_curve(px, py, names, save_dir, x_label="Confidence", y_label="Metric"):
    from matplotlib import pyplot

    figure, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), f=0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.3f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{y_label}-Confidence Curve")
    figure.savefig(save_dir, dpi=250)
    pyplot.close(figure)


def compute_ap(tp, conf, output, target, plot=False, names=(), eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(start=0, stop=1, num=1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
            if plot and j == 0:
                py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    if plot:
        names = dict(enumerate(names))  # to dict
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        plot_pr_curve(px, py, ap, names, save_dir="./weights/PR_curve.png")
        plot_curve(px, f1, names, save_dir="./weights/F1_curve.png", y_label="F1")
        plot_curve(px, p, names, save_dir="./weights/P_curve.png", y_label="Precision")
        plot_curve(px, r, names, save_dir="./weights/R_curve.png", y_label="Recall")
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def compute_iou(box1, box2, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU


def strip_optimizer(filename):
    x = torch.load(filename, map_location="cpu")
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f=filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().cpu()

    ckpt = {}
    for k, v in src.state_dict().items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v

    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # norm-weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


class CosineLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps))

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class Assigner(torch.nn.Module):
    def __init__(self, nc=80, top_k=13, alpha=1.0, beta=6.0, eps=1E-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)

        if num_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        num_anchors = anc_points.shape[0]
        shape = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        mask_in_gts = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = mask_in_gts.view(shape[0], shape[1], num_anchors, -1).amin(3).gt_(self.eps)
        na = pd_bboxes.shape[-2]
        gt_mask = (mask_in_gts * mask_gt).bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, batch_size, num_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=batch_size).view(-1, 1).expand(-1, num_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        bbox_scores[gt_mask] = pd_scores[ind[0], :, ind[1]][gt_mask]  # b, max_num_obj, h*w

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[gt_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[gt_mask]
        overlaps[gt_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        mask_top_k = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k):
            mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k.masked_fill_(mask_top_k > 1, 0)
        mask_top_k = mask_top_k.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)

        # Assigned target
        index = torch.arange(end=batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_index = target_gt_idx + index * num_max_boxes
        target_labels = gt_labels.long().flatten()[target_index]

        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_index]

        # Assigned target scores
        target_labels.clamp_(0)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                    dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()


class QFL(torch.nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        return torch.pow(torch.abs(targets - outputs.sigmoid()), self.beta) * bce_loss


class VFL(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.00, iou_weighted=True):
        super().__init__()
        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        assert outputs.size() == targets.size()
        targets = targets.type_as(outputs)

        if self.iou_weighted:
            focal_weight = targets * (targets > 0.0).float() + \
                           self.alpha * (outputs.sigmoid() - targets).abs().pow(self.gamma) * \
                           (targets <= 0.0).float()

        else:
            focal_weight = (targets > 0.0).float() + \
                           self.alpha * (outputs.sigmoid() - targets).abs().pow(self.gamma) * \
                           (targets <= 0.0).float()

        return self.bce_loss(outputs, targets) * focal_weight


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        loss = self.bce_loss(outputs, targets)

        if self.alpha > 0:
            alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss *= alpha_factor

        if self.gamma > 0:
            outputs_sigmoid = outputs.sigmoid()
            p_t = targets * outputs_sigmoid + (1 - targets) * (1 - outputs_sigmoid)
            gamma_factor = (1.0 - p_t) ** self.gamma
            loss *= gamma_factor

        return loss


class BoxLoss(torch.nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self.df_loss(pred_dist[fg_mask].view(-1, self.dfl_ch + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_box, loss_dfl

    @staticmethod
    def df_loss(pred_dist, target):
        # Distribution Focal Loss (DFL)
        # https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        left_loss = cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class ComputeLoss:
    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device

        m = model.head  # Head() module

        self.params = params
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.ch
        self.device = device

        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)

        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    def __call__(self, outputs, targets):
        x = torch.cat([i.view(outputs[0].shape[0], self.no, -1) for i in outputs], dim=2)
        pred_distri, pred_scores = x.split(split_size=(self.reg_max * 4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)

        idx = targets['idx'].view(-1, 1)
        cls = targets['cls'].view(-1, 1)
        box = targets['box']

        targets = torch.cat((idx, cls, box), dim=1).to(self.device)
        if targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2  # half-width
            dh = x[..., 3] / 2  # half-height
            y[..., 0] = x[..., 0] - dw  # top left x
            y[..., 1] = x[..., 1] - dh  # top left y
            y[..., 2] = x[..., 0] + dw  # bottom right x
            y[..., 3] = x[..., 1] + dh  # bottom right y
            gt[..., 1:5] = y
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        assigned_targets = self.assigner(pred_scores.detach().sigmoid(),
                                         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                         anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask = assigned_targets

        target_scores_sum = max(target_scores.sum(), 1)

        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum  # BCE

        # Box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(pred_distri,
                                               pred_bboxes,
                                               anchor_points,
                                               target_bboxes,
                                               target_scores,
                                               target_scores_sum, fg_mask)

        loss_box *= self.params['box']  # box gain
        loss_cls *= self.params['cls']  # cls gain
        loss_dfl *= self.params['dfl']  # dfl gain

        return loss_box, loss_cls, loss_dfl


# -----------------------------------------------------------------
# DeepSORT: Util 2
# -----------------------------------------------------------------


def _pair_distance(a, b):
    
    a, b = numpy.asarray(a), numpy.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return numpy.zeros((len(a), len(b)))
    a2, b2 = numpy.square(a).sum(axis=1), numpy.square(b).sum(axis=1)
    r2 = -2. * numpy.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = numpy.clip(r2, 0., float(numpy.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    
    if not data_is_normalized:
        a = numpy.asarray(a) / numpy.linalg.norm(a, axis=1, keepdims=True)
        b = numpy.asarray(b) / numpy.linalg.norm(b, axis=1, keepdims=True)
    return 1. - numpy.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    distances = _pair_distance(x, y)
    return numpy.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


def iou(bbox, candidates):
    
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = numpy.c_[numpy.maximum(bbox_tl[0], candidates_tl[:, 0])[:, numpy.newaxis],
    numpy.maximum(bbox_tl[1], candidates_tl[:, 1])[:, numpy.newaxis]]
    br = numpy.c_[numpy.minimum(bbox_br[0], candidates_br[:, 0])[:, numpy.newaxis],
    numpy.minimum(bbox_br[1], candidates_br[:, 1])[:, numpy.newaxis]]
    wh = numpy.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    
    if track_indices is None:
        track_indices = numpy.arange(len(tracks))
    if detection_indices is None:
        detection_indices = numpy.arange(len(detections))

    cost_matrix = numpy.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = numpy.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def min_cost_matching(distance_metric, max_distance, tracks,
                      detections, track_indices=None, detection_indices=None):
    
    if track_indices is None:
        track_indices = numpy.arange(len(tracks))
    if detection_indices is None:
        detection_indices = numpy.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth,
                     tracks, detections, track_indices=None, detection_indices=None):
    
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [k for k in track_indices
                           if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = min_cost_matching(distance_metric,
                                                               max_distance, tracks, detections,
                                                               track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices,
                     detection_indices, gated_cost=INFTY_COST, only_position=False):
    
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = numpy.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance,
                                             measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


class NearestNeighborDistanceMetric(object):
    
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        
        cost_matrix = numpy.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class KalmanFilter(object):
    
    def __init__(self):
        ndim, dt = 4, 1.

        self._motion_mat = numpy.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = numpy.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        
        mean_pos = measurement
        mean_vel = numpy.zeros_like(mean_pos)
        mean = numpy.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        covariance = numpy.diag(numpy.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = numpy.diag(numpy.square(numpy.r_[std_pos, std_vel]))

        mean = numpy.dot(self._motion_mat, mean)
        covariance = numpy.linalg.multi_dot((self._motion_mat,
                                             covariance,
                                             self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = numpy.diag(numpy.square(std))

        mean = numpy.dot(self._update_mat, mean)
        covariance = numpy.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = linalg.cho_solve((chol_factor, lower),
                                       numpy.dot(covariance, self._update_mat.T).T,
                                       check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + numpy.dot(innovation, kalman_gain.T)
        new_covariance = covariance - numpy.linalg.multi_dot((kalman_gain,
                                                              projected_cov,
                                                              kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = numpy.linalg.cholesky(covariance)
        d = measurements - mean
        z = linalg.solve_triangular(cholesky_factor,
                                    d.T, lower=True,
                                    check_finite=False, overwrite_b=True)
        squared_maha = numpy.sum(z * z, axis=0)
        return squared_maha


class TrackState:
    
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    
    def __init__(self, mean, covariance, track_id, n_init, max_age, oid, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.oid = oid

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()

    def update(self, kf, detection):
        
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


class Detection(object):
    
    def __init__(self, tlwh, confidence, feature, oid):
        self.tlwh = numpy.asarray(tlwh, dtype=numpy.float64)
        self.confidence = float(confidence)
        self.feature = numpy.asarray(feature, dtype=numpy.float32)
        self.oid = oid

    def to_tlbr(self):
        
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class Tracker:
    
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(numpy.asarray(features), numpy.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = numpy.array([dets[i].feature for i in detection_indices])
            targets = numpy.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(self.kf, cost_matrix, tracks,
                                           dets, track_indices, detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(gated_metric,
                                                                               self.metric.matching_threshold,
                                                                               self.max_age,
                                                                               self.tracks, detections,
                                                                               confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if
                                                     self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(iou_cost, self.max_iou_distance, self.tracks,
                              detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, detection.oid,
            detection.feature))
        self._next_id += 1