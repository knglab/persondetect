import cv2
import numpy as np
import tensorflow as tf

INT2CLASS_NAME = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def prepare_image(img):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    # Convert from BGR to RGB
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # Resize and pad the image
    # this config of letterbox use only for yolov5m-fp16.tflite, for other config, you can use detect.py in master 
    # branch to find out the config
    img, ratio, (dw, dh) = letterbox(img,new_shape=(640,640),stride=32,auto=False)
    # convert to float
    img = img.astype(np.float32)
    # Normalize from [0, 255] to [0, 1]
    img /= 255.0
    # Expand the image to batch
    img = np.expand_dims(img, 0)
    return img, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def int2str(x):
    """
    Convert index to class name.

    Returns a string
    """
    return INT2CLASS_NAME[x]

def scale_coords(img_shape, coords,ratio,padding):
    # Rescale coords (xyxy) from img_shape to img_raw_shape
    height, width = img_shape
    coords[:, [0, 2]] =(coords[:, [0, 2]] * width  - padding[0]) / ratio[0]  # x padding
    coords[:, [1, 3]] =(coords[:, [1, 3]] * height  - padding[1]) / ratio[1]  # y padding
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]],0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]],0, shape[0])  # y1, y2
    return boxes


def decode_prediction(prediction,conf_thres=0.25,iou_thres = 0.45,max_output_size = 100):
    # Ignore low confidence predictions
    xc = prediction[..., 4] > conf_thres  # candidates
    # Retrieve obj indices have object confidence > conf_thres
    candidates = prediction[xc]  # confidence
    # Compute confidences
    candidates[:, 5:] *= candidates[:, 4:5]  # conf = obj_conf * cls_conf
    # Convert bbox coordinates to xyxy
    bboxes = xywh2xyxy(candidates[:, :4])
    # Get class index
    class_index = np.argmax(candidates[:,5:], axis=1)
    # Get class probablities
    class_probs = np.take_along_axis(candidates[:,5:],np.expand_dims(class_index,axis=1),axis=1)
    class_index = np.expand_dims(class_index,axis=1)
    # Merge predictions, scores, and class index
    predictions = np.concatenate([bboxes,class_probs,class_index],axis=1)
    # Non maximum suppression
    selected_indices = tf.image.non_max_suppression(
      bboxes, class_probs.squeeze(axis=1), max_output_size, iou_thres)
    predictions = tf.gather(predictions, selected_indices).numpy()
    return predictions







