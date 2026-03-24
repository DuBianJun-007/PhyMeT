import numpy as np
import torch
from torchvision.ops import nms, boxes


def _single_scale_nms(prediction, conf_thres, nms_iou, num_classes):
    """
    单尺度NMS处理

    Args:
        prediction: [N, 4+1+num_classes] 单尺度预测张量
        conf_thres: 置信度阈值
        nms_iou: NMS IoU阈值
        num_classes: 类别数

    Returns:
        检测结果张量 [M, 7] (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        如果没有检测，返回None
    """
    if prediction.numel() == 0:
        return None

    prediction = prediction.clone()

    class_conf, class_pred = torch.max(
        prediction[:, :, 5 : 5 + num_classes], 2, keepdim=True
    )
    obj_conf = prediction[:, :, 4:5]
    conf_mask = (obj_conf * class_conf >= conf_thres).squeeze(-1)

    if not conf_mask.any():
        return None

    prediction = prediction[conf_mask]
    class_conf = class_conf[conf_mask]
    class_pred = class_pred.squeeze(-1)
    obj_conf = obj_conf[conf_mask]

    if prediction.numel() == 0:
        return None

    box_xywh = prediction[:, :4]
    box_xyxy = torch.zeros_like(box_xywh)
    box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2
    box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2
    box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2
    box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2

    total_conf = (obj_conf * class_conf).squeeze(-1)

    if box_xyxy.numel() == 0:
        return None

    nms_out_index = boxes.batched_nms(
        box_xyxy,
        total_conf,
        class_pred.long(),
        nms_iou,
    )

    detections = torch.cat(
        [
            box_xyxy[nms_out_index],
            obj_conf[nms_out_index],
            class_conf[nms_out_index],
            class_pred[nms_out_index].float().unsqueeze(-1),
        ],
        dim=1,
    )

    return detections


def _merge_multi_scale_results(detections_list):
    """
    合并多尺度NMS结果

    Args:
        detections_list: 各尺度的检测结果列表

    Returns:
        合并后的检测结果 tensor，如果没有检测返回None
    """
    valid_detections = [d for d in detections_list if d is not None]

    if not valid_detections:
        return None

    return torch.cat(valid_detections, dim=0)


def yolo_correct_boxes(
    box_xy, box_wh, input_shape, image_shape, letterbox_image, pad_info=None
):
    """
    把预测框坐标转换到原始图像尺寸
    支持边缘padding还原

    Args:
        box_xy: 中心点坐标
        box_wh: 宽高
        input_shape: 模型输入尺寸 (w, h)
        image_shape: 原始图像尺寸 (h, w)
        letterbox_image: 是否使用letterbox
        pad_info: padding信息字典，包含pad_left, pad_top, scale等
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    if pad_info is not None:
        scale = pad_info.get("scale", 1.0)
        pad_left = pad_info.get("pad_left", 0)
        pad_top = pad_info.get("pad_top", 0)

        box_yx = (box_yx - np.array([pad_left, pad_top]) / input_shape[::-1]) / scale
        box_hw = box_hw / scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = np.concatenate(
        [
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2],
        ],
        axis=-1,
    )
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def decode_outputs(outputs, input_shape):
    grids = []
    strides = []
    hw = [x.shape[-2:] for x in outputs]
    # ---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    # ---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(
        0, 2, 1
    )
    # ---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    # ---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        # ---------------------------#
        #   根据特征层的高宽生成网格点
        # ---------------------------#
        grid_y, grid_x = torch.meshgrid(
            [torch.arange(h), torch.arange(w)], indexing="ij"
        )
        # ---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        # ---------------------------#
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    # ---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    # ---------------------------#
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    # ------------------------#
    #   根据网格点进行解码
    # ------------------------#
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # -----------------#
    #   归一化
    # -----------------#
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def non_max_suppression(
    prediction,
    num_classes,
    input_shape,
    image_shape,
    letterbox_image,
    conf_thres=0.5,
    nms_thres=0.4,
):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # ----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )

        output[i] = detections[nms_out_index]

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (
                (output[i][:, 0:2] + output[i][:, 2:4]) / 2,
                output[i][:, 2:4] - output[i][:, 0:2],
            )
            output[i][:, :4] = yolo_correct_boxes(
                box_xy, box_wh, input_shape, image_shape, letterbox_image
            )
    return output


def non_max_suppression_multi_scale(
    prediction_list,
    num_classes,
    input_shape,
    image_shape,
    letterbox_image,
    conf_thres=0.5,
    nms_thres=0.4,
    pad_info=None,
):
    """
    多尺度分别NMS处理

    Args:
        prediction_list: [P3_pred, P4_pred, P5_pred] 多尺度预测列表
        num_classes: 类别数
        input_shape: 模型输入尺寸 (w, h)
        image_shape: 原始图像尺寸 (h, w)
        letterbox_image: 是否使用letterbox
        conf_thres: 置信度阈值
        nms_thres: NMS IoU阈值
        pad_info: padding信息字典，用于边缘padding还原

    Returns:
        output: [batch_size] 每个图像的检测结果列表
    """
    batch_size = prediction_list[0].shape[0]
    output = [None for _ in range(batch_size)]

    for batch_idx in range(batch_size):
        scale_detections = []

        for prediction in prediction_list:
            image_pred = prediction[batch_idx]

            if image_pred.numel() == 0:
                continue

            if image_pred.dim() == 3:
                image_pred = image_pred.flatten(start_dim=1)

            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )
            obj_conf = image_pred[:, 4:5]

            total_conf = obj_conf * class_conf
            conf_mask = total_conf.squeeze(-1) >= conf_thres

            if not conf_mask.any():
                continue

            valid_indices = conf_mask.nonzero(as_tuple=True)[0]

            if valid_indices.numel() == 0:
                continue

            filtered_pred = image_pred[valid_indices]
            filtered_obj = obj_conf[valid_indices]
            filtered_class_conf = class_conf[valid_indices]
            filtered_class_pred = class_pred[valid_indices]

            box_xywh = filtered_pred[:, :4]
            box_xyxy = torch.zeros_like(box_xywh)
            box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2
            box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2
            box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2
            box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2

            total_conf_filtered = (filtered_obj * filtered_class_conf).squeeze(-1)

            if box_xyxy.numel() == 0:
                continue

            nms_out_index = boxes.batched_nms(
                box_xyxy,
                total_conf_filtered,
                filtered_class_pred.squeeze(-1).long(),
                nms_thres,
            )

            detections = torch.cat(
                [
                    box_xyxy[nms_out_index],
                    filtered_obj[nms_out_index],
                    filtered_class_conf[nms_out_index],
                    filtered_class_pred[nms_out_index].float(),
                ],
                dim=1,
            )
            scale_detections.append(detections)

        if not scale_detections:
            output[batch_idx] = None
            continue

        all_detections = torch.cat(scale_detections, dim=0)

        if all_detections.numel() == 0:
            output[batch_idx] = None
            continue

        all_detections = all_detections.cpu().numpy()

        box_xy = (all_detections[:, 0:2] + all_detections[:, 2:4]) / 2
        box_wh = all_detections[:, 2:4] - all_detections[:, 0:2]
        all_detections[:, :4] = yolo_correct_boxes(
            box_xy, box_wh, input_shape, image_shape, letterbox_image, pad_info
        )

        output[batch_idx] = all_detections

    return output
