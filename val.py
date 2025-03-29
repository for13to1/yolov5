# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlpackage          # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """
    Saves one detection result to a txt file in normalized xywh format, optionally including confidence.

    Args:
        predn (torch.Tensor): Predicted bounding boxes and associated confidence scores and classes in xyxy format, tensor
            of shape (N, 6) where N is the number of detections.
        save_conf (bool): If True, saves the confidence scores along with the bounding box coordinates.
        shape (tuple): Shape of the original image as (height, width).
        file (str | Path): File path where the result will be saved.

    Returns:
        None

    Notes:
        The xyxy bounding box format represents the coordinates (xmin, ymin, xmax, ymax).
        The xywh format represents the coordinates (center_x, center_y, width, height) and is normalized by the width and
        height of the image.

    Example:
        ```python
        predn = torch.tensor([[10, 20, 30, 40, 0.9, 1]])  # example prediction
        save_one_txt(predn, save_conf=True, shape=(640, 480), file="output.txt")
        ```
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    """
    Saves a single JSON detection result, including image ID, category ID, bounding box, and confidence score.

    Args:
        predn (torch.Tensor): Predicted detections in xyxy format with shape (n, 6) where n is the number of detections.
                              The tensor should contain [x_min, y_min, x_max, y_max, confidence, class_id] for each detection.
        jdict (list[dict]): List to collect JSON formatted detection results.
        path (pathlib.Path): Path object of the image file, used to extract image_id.
        class_map (dict[int, int]): Mapping from model class indices to dataset-specific category IDs.

    Returns:
        None: Appends detection results as dictionaries to `jdict` list in-place.

    Example:
        ```python
        predn = torch.tensor([[100, 50, 200, 150, 0.9, 0], [50, 30, 100, 80, 0.8, 1]])
        jdict = []
        path = Path("42.jpg")
        class_map = {0: 18, 1: 19}
        save_one_json(predn, jdict, path, class_map)
        ```
        This will append to `jdict`:
        ```
        [
            {'image_id': 42, 'category_id': 18, 'bbox': [125.0, 75.0, 100.0, 100.0], 'score': 0.9},
            {'image_id': 42, 'category_id': 19, 'bbox': [75.0, 55.0, 50.0, 50.0], 'score': 0.8}
        ]
        ```

    Notes:
        The `bbox` values are formatted as [x, y, width, height], where x and y represent the top-left corner of the box.
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format
            [x1, y1, x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true positive
            for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Example:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data, #! Path to a dataset YAML file or a dataset dictionary
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):
    """
    Evaluates a YOLOv5 model on a dataset and logs performance metrics.

    Args:
        data (str | dict): Path to a dataset YAML file or a dataset dictionary.
        weights (str | list[str], optional): Path to the model weights file(s). Supports various formats including PyTorch,
            TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel, TensorFlow GraphDef, TensorFlow Lite,
            TensorFlow Edge TPU, and PaddlePaddle.
        batch_size (int, optional): Batch size for inference. Default is 32.
        imgsz (int, optional): Input image size (pixels). Default is 640.
        conf_thres (float, optional): Confidence threshold for object detection. Default is 0.001.
        iou_thres (float, optional): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.6.
        max_det (int, optional): Maximum number of detections per image. Default is 300.
        task (str, optional): Task type - 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        device (str, optional): Device to use for computation, e.g., '0' or '0,1,2,3' for CUDA or 'cpu' for CPU. Default is ''.
        workers (int, optional): Number of dataloader workers. Default is 8.
        single_cls (bool, optional): Treat dataset as a single class. Default is False.
        augment (bool, optional): Enable augmented inference. Default is False.
        verbose (bool, optional): Enable verbose output. Default is False.
        save_txt (bool, optional): Save results to *.txt files. Default is False.
        save_hybrid (bool, optional): Save label and prediction hybrid results to *.txt files. Default is False.
        save_conf (bool, optional): Save confidences in --save-txt labels. Default is False.
        save_json (bool, optional): Save a COCO-JSON results file. Default is False.
        project (str | Path, optional): Directory to save results. Default is ROOT/'runs/val'.
        name (str, optional): Name of the run. Default is 'exp'.
        exist_ok (bool, optional): Overwrite existing project/name without incrementing. Default is False.
        half (bool, optional): Use FP16 half-precision inference. Default is True.
        dnn (bool, optional): Use OpenCV DNN for ONNX inference. Default is False.
        model (torch.nn.Module, optional): Model object for training. Default is None.
        dataloader (torch.utils.data.DataLoader, optional): Dataloader object. Default is None.
        save_dir (Path, optional): Directory to save results. Default is Path('').
        plots (bool, optional): Plot validation images and metrics. Default is True.
        callbacks (utils.callbacks.Callbacks, optional): Callbacks for logging and monitoring. Default is Callbacks().
        compute_loss (function, optional): Loss function for training. Default is None.

    Returns:
        dict: Contains performance metrics including precision, recall, mAP50, and mAP50-95.
    """

    # Initialize/load model and set device
    training = model is not None
    #! 若 model 存在，则 training 被设置为 True；若 model 不存在，则 training 被设置为 False：什么意思？
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        #? batch_size 指的是一次处理的数据样本数量。例如，如果 batch_size 为 32，那么模型会一次处理 32 个数据样本，然后更新权重，接着再处理下一个批量的 32 个数据样本。
        #! 返回值是 torch.device 对象，表示计算设备的类型，例如 "cpu" 或 "cuda:0"。

        # Directories
        #! project 默认为 ROOT / "runs/val"
        #! name 默认为 exp
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        #! 若 save_txt 为 True，则 save_dir / "labels" 会被创建
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        #? stride: 如果 stride 为 2，那么卷积核将每次移动 2 个像素。这样可以减少输出特征图的大小，同时保留输入数据的重要信息。
        #? pt: PyTorch model
        #? jit: TorchScript model
        #? engine: TensorRT engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        #? model.fp16 表示模型是否支持 FP16 计算。如果模型支持 FP16，则 half 变量将被设置为 True，否则将被设置为 False
        #! FP16 计算仅在某些后端（backend）上支持，并且需要 CUDA（一种 NVIDIA 的并行计算平台）。
        if engine:
            #? 优化引擎（如 TensorRT）：batch_size 需在导出时预先设定，推理时不可动态修改
            batch_size = model.batch_size
        else:
            device = model.device #! 这里返回的可能是什么类型？
            if not (pt or jit):
                #! 若不支持 PyTorch 或 TorchScript，则将 batch_size 设置为 1
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  #! 返回的是字典

    # Configure
    model.eval() #! 将模型切换到评估模式（evaluation mode）
    #? 在 PyTorch 中，模型有两种模式：训练模式（training mode）和评估模式（evaluation mode）。训练模式用于训练模型，评估模式用于测试和评估模型。当模型处于训练模式时，它会计算梯度并更新权重。然而，在评估模式下，模型不会计算梯度，也不会更新权重。
    cuda = device.type != "cpu"
    #? https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
    #! 不是 cpu 的不一定是 cuda
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    #! data["val"] 是字符串且以 coco/val2017.txt 结尾，则 is_coco 为 True
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    #! 如果 single_cls 为 True，则 nc 为 1，否则 nc 为 data["nc"] 的值
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    #! [0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500]
    #! COCO数据集的评估标准，要求模型在IoU阈值从0.5到0.95（步长0.05）的范围内均表现良好
    niou = iouv.numel() # 10

    # Dataloader
    if not training: #! 仅在非训练模式下执行（验证/推理）
        if pt and not single_cls:  # check --weights are trained on --data
            #! 支持 PyTorch，且非单类别模式
            ncm = model.model.nc #! 使用模型的 nc 值，即类别数
            #! 预训练权重的类别数必须与数据集的类别数一致（除非显式启用单类别模式）
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )

        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        #! 若 pt 为 True，则 imgsz 为 (1, 3, imgsz, imgsz)，否则 imgsz 为 (batch_size, 3, imgsz, imgsz)

        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        #! task == "speed": 不填充，拉伸为正方形（容易引入形变）: 最大化吞吐量，适合测速
        #! rectangle inference: 等比例缩放+填充到正方形

        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images

        #! 函数返回的是 (dataloader, dataset), 这里只取 dataloader
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)

    # get class names
    names = model.names if hasattr(model, "names") else model.module.names
    #? 兼容两种常见场景：
    #? 单设备（如单 GPU 或 CPU）模型：直接访问 model.names。
    #? 多设备（如多 GPU DataParallel 或分布式 DistributedDataParallel）模型：通过 model.module.names 访问。
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))

    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    #? 将YOLO训练时使用的80个类别索引映射回COCO官方91个类别的原始ID: COCO数据集实际有91个类别，但部分类别因样本少被合并，训练时仅用80类。此函数确保评估结果与官方类别ID对齐

    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    #? tp: true positive
    #? fp: false positive
    #? p: precision
    #? r: recall
    #? f1: F1 score
    #? mp: mean precision
    #? mr: mean recall
    #? map50: mean average precision at 0.5
    #? ap50: average precision at 0.5
    #? map: mean average precision

    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times

    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    #! jdict: List to collect JSON formatted detection results.
    callbacks.run("on_val_start")

    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        #! 逐批次地
        callbacks.run("on_val_batch_start")
        #? 触发验证批次开始时的回调钩子

        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                #? non_blocking=True: 异步传输图像数据
                targets = targets.to(device)
                #? non_blocking=False: 同步传输标签数据
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            #? compute_loss: 是否保留训练输出（用于损失计算）
            #? augment: 推理时应用数据增强（如多尺度/翻转）
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
            #! 模型本批次输出结果的 shape 为 (batch_size, anchor_num, 5+mask_param_num+class_num)
            #! 其中 5 对应 (x_center, y_center, width, height, obj_confidence)
            #? model(im): 是PyTorch的语法糖，自动触发前向传播
            #? preds：模型的预测结果（边界框、类别、置信度）
            #? train_out：训练所需的中间输出（如各层特征），用于后续损失计算
            #! compute_loss=False 时，启用数据增强（如测试时增强，TTA），但仅返回预测结果 preds

        # Loss
        if compute_loss:
            #! 接收模型输出 train_out 和真实标签 targets，返回损失值
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        #? targets 的 shape 为 (object_num, 6)
        #? targets 的格式为 [batch_index, class_id, x_center, y_center, width, height]，其中坐标是归一化的（范围[0,1]），乘以图像的宽（width）和高（height），将归一化坐标转换为实际像素值
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        #? save_hybrid: 控制是否启用混合模式（如同时使用标注数据和自动生成标签）
        #? targets[:, 0] == i: 用于获取第 i 张图上的所有 targets
        #? targets[targets[:, 0] == i, 1:]: 获取第 i 张图上的所有 targets 中的 class_id、x_center、y_center、width、height
        #! lb: 参考标签

        # NMS
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )
            #! multi_label=True: 单个框可以预测多个类别标签
            #! agnostic=single_cls: 若为True，则跨类别进行NMS（适用于单类别任务）
            #! max_det: 每个图像最多保留的预测框数
            #! 返回值 preds 的 shape 为长度为 batch_size 的列表，每个元素的 shape 为 (max_det, 6): 其中 6 对应 (x1, y1, x2, y2, conf, cls)

        # Metrics
        for si, pred in enumerate(preds): #! si: sequence index
            labels = targets[targets[:, 0] == si, 1:]
            #? targets[:, 0] == si: 用于获取第 si 张图上的所有 targets
            #? targets[targets[:, 0] == si, 1:]: 获取第 si 张图上的所有 targets 中的 class_id、x_center、y_center、width、height
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            #? nl: 第 si 张图上的 labels 的数量
            #? npr: 第 si 张图上的预测框的数量
            path, shape = Path(paths[si]), shapes[si][0]
            #? path: 第 si 张图的路径
            #? shape: 第 si 张图的原始尺寸
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            #? correct[i, j] = True 表示第 i 个预测框在IoU阈值 iouv[j] 下正确匹配某真实框
            #? 若预测框与某真实框的 IoU > 阈值 且类别正确，标记对应位置为 True
            seen += 1 #? 已处理图像计数+1

            if npr == 0: #? 所有预测框的置信度低于阈值或在NMS中被过滤，导致无有效检测结果
                if nl: #? 若存在真实目标但无预测框，需记录漏检信息
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    #? labels[:, 0]：真实标签的类别ID（用于统计各类别的漏检情况）
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        #? detections=None 表示无预测框，labels 提供真实类别
                        #! 所有真实标签被计为假阴性（FN），更新混淆矩阵的FN计数
                continue
                #! 无预测框时无需执行坐标转换、置信度计算等步骤，直接处理下一张图像

            # Predictions
            if single_cls:
                #! 当启用单类别模式（single_cls=True）时，将所有预测框的类别ID强制设为 0
                pred[:, 5] = 0
            predn = pred.clone()
            #! 将预测框坐标从 模型输入空间（如填充后的640x640）映射回 原始图像空间（如1280x720），考虑可能的缩放和填充
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            #? im[si].shape[1:]：模型输入图像的尺寸（如 (640, 640)），即预处理后的尺寸
            #? predn[:, :4]：待转换的预测框坐标（x1, y1, x2, y2），归一化到输入图像尺寸
            #? shape：原始图像的尺寸（如 (1280, 720)）
            #? shapes[si][1]：预处理时的缩放比例或填充信息，用于逆向计算

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                #! labels[:, 1:5] 形状为 [num_labels, 4]，包含归一化的中心坐标和宽高
                #! tbox 形状为 [num_labels, 4]，转换为归一化的角点坐标 [x1, y1, x2, y2]
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                #! im[si].shape[1:]：模型输入图像的尺寸（如 (640, 640)）
                #! tbox：待转换的真实标签坐标（归一化到输入图像尺寸）
                #! shape：原始图像的尺寸（如 (1280, 720)）
                #! shapes[si][1]: 预处理时的缩放比例和填充信息
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                #! labelsn 形状为 [num_labels, 5]，格式为 [class_id, x1, y1, x2, y2]
                correct = process_batch(predn, labelsn, iouv)
                #! 逐个预测框与真实标签计算IoU，判断是否在不同阈值下匹配成功
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
                    #! 记录预测框与真实标签的类别匹配情况，生成分类性能分析图
                    #! 预测框与真实框IoU > 阈值且类别正确 → 真阳性（TP）
                    #! 预测框未匹配任何真实框 → 假阳性（FP）
                    #! 真实框未被任何预测框匹配 → 假阴性（FN）

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            #! correct：匹配矩阵，形状 [num_pred, num_iou_thresholds]
            #! pred[:, 4]：预测框的置信度，形状 [num_pred]
            #! pred[:, 5]：预测的类别概率，形状 [num_pred]
            #! labels[:, 0]：真实标签的类别ID，形状 [num_labels]

            # Save/log
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            #? targets: 当前批次的目标框
            #? preds: 当前批次的预测框
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """
    Parse command-line options for configuring YOLOv5 model inference.

    Args:
        data (str, optional): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        weights (list[str], optional): List of paths to model weight files. Default is 'yolov5s.pt'.
        batch_size (int, optional): Batch size for inference. Default is 32.
        imgsz (int, optional): Inference image size in pixels. Default is 640.
        conf_thres (float, optional): Confidence threshold for predictions. Default is 0.001.
        iou_thres (float, optional): IoU threshold for Non-Max Suppression (NMS). Default is 0.6.
        max_det (int, optional): Maximum number of detections per image. Default is 300.
        task (str, optional): Task type - options are 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        device (str, optional): Device to run the model on. e.g., '0' or '0,1,2,3' or 'cpu'. Default is empty to let the system choose automatically.
        workers (int, optional): Maximum number of dataloader workers per rank in DDP mode. Default is 8.
        single_cls (bool, optional): If set, treats the dataset as a single-class dataset. Default is False.
        augment (bool, optional): If set, performs augmented inference. Default is False.
        verbose (bool, optional): If set, reports mAP by class. Default is False.
        save_txt (bool, optional): If set, saves results to *.txt files. Default is False.
        save_hybrid (bool, optional): If set, saves label+prediction hybrid results to *.txt files. Default is False.
        save_conf (bool, optional): If set, saves confidences in --save-txt labels. Default is False.
        save_json (bool, optional): If set, saves results to a COCO-JSON file. Default is False.
        project (str, optional): Project directory to save results to. Default is 'runs/val'.
        name (str, optional): Name of the directory to save results to. Default is 'exp'.
        exist_ok (bool, optional): If set, existing directory will not be incremented. Default is False.
        half (bool, optional): If set, uses FP16 half-precision inference. Default is False.
        dnn (bool, optional): If set, uses OpenCV DNN for ONNX inference. Default is False.

    Returns:
        argparse.Namespace: Parsed command-line options.

    Notes:
        - The '--data' parameter is checked to ensure it ends with 'coco.yaml' if '--save-json' is set.
        - The '--save-txt' option is set to True if '--save-hybrid' is enabled.
        - Args are printed using `print_args` to facilitate debugging.

    Example:
        To validate a trained YOLOv5 model on a COCO dataset:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
        Different model formats could be used instead of `yolov5s.pt`:
        ```python
        $ python val.py --weights yolov5s.pt yolov5s.torchscript yolov5s.onnx yolov5s_openvino_model yolov5s.engine
        ```
        Additional options include saving results in different formats, selecting devices, and more.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="mps", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 tasks like training, validation, testing, speed, and study benchmarks based on provided options.

    Args:
        opt (argparse.Namespace): Parsed command-line options.
            This includes values for parameters like 'data', 'weights', 'batch_size', 'imgsz', 'conf_thres',
            'iou_thres', 'max_det', 'task', 'device', 'workers', 'single_cls', 'augment', 'verbose', 'save_txt',
            'save_hybrid', 'save_conf', 'save_json', 'project', 'name', 'exist_ok', 'half', and 'dnn', essential
            for configuring the YOLOv5 tasks.

    Returns:
        None

    Examples:
        To validate a trained YOLOv5 model on the COCO dataset with a specific weights file, use:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
