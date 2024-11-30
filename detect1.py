# Ultralytics YOLOv5 🚀, AGPL-3.0 license
# 실행 방법 :
# python detect1.py --weights runs/train/exp3/weights/best.pt --source pic/pic/*.jpg --save-crop 
#
# python detect1.py --source pic/pic/*.jpg --save-txt --weights runs/train/exp2/weights/best.pt --conf 0.6      #학습용

"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
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
import csv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   #기기에 설치된 소프트웨어적인 문제로 인한 임시조치
import platform
import sys
from pathlib import Path

import torch

#openCV, OCR
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = Path(project)/'result'
    #save_dir.mkdir(parents=True, exist_ok=True)
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # im.jpg
            save_path = str(save_dir / 'Labels' / p.stem) + f'_{frame}' + '.jpg'
            txt_path = str(save_dir / "labels" / p.stem) + f'_{frame}'
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        #annotator.box_label(xyxy, label, color=colors(c, True)) #탐지부분 라벨 처리
                    if save_crop:
                        
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"Enter--{timestamp}.jpg", BGR=True)
                        #save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                        ##################################################################################################################################
                        #                                                           번호판 출력 구역                                                      #
                        ##################################################################################################################################


                        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

                        #이미지 읽어오기

                        """
                        - 특정 folder 내에 있는 "가장 최근에 생성된" 파일을 리턴하는 방법 
                        """
                        folder_path = 'HBOS3/runs/detect/result/crops/license_plate/'

                        # each_file_path_and_gen_time: 각 file의 경로와, 생성 시간을 저장함
                        each_file_path_and_gen_time = []
                        for each_file_name in os.listdir(folder_path):
                            # getctime: 입력받은 경로에 대한 생성 시간을 리턴
                            each_file_path = folder_path + each_file_name
                            each_file_gen_time = os.path.getctime(each_file_path)
                            each_file_path_and_gen_time.append(
                                (each_file_path, each_file_gen_time)
                            )

                        # 가장 생성시각이 큰(가장 최근인) 파일을 리턴 
                        most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
                        img_ori = cv2.imread(most_recent_file)

                        height, width, channel = img_ori.shape

                        #그레이 스케일로 변환
                        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)




                        #Adaptive Thresholding

                        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

                        img_blur_thresh = cv2.adaptiveThreshold(
                            img_blurred,
                            maxValue=255.0,
                            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            thresholdType=cv2.THRESH_BINARY_INV,
                            blockSize=19,
                            C=9
                        )


                        # 윤곽선 찾기
                        contours, _ = cv2.findContours(
                            img_blur_thresh,
                            mode=cv2.RETR_LIST,
                            method=cv2.CHAIN_APPROX_SIMPLE
                        )


                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
                        #데이터 준비
                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        contours_dict = []

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
                            
                            contours_dict.append({
                                'contour': contour,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h,
                                'cx': x + (w / 2),
                                'cy': y + (h / 2)
                            })
                            

                        #Select Candidates by Char Size
                        MIN_AREA = 80
                        MIN_WIDTH, MIN_HEIGHT=2, 8
                        MIN_RATIO, MAX_RATIO = 0.25, 1.0

                        possible_contours = []

                        cnt = 0
                        for d in contours_dict:
                            area = d['w'] * d['h']
                            ratio = d['w'] / d['h']
                            
                            if area > MIN_AREA \
                            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                            and MIN_RATIO < ratio < MAX_RATIO:
                                d['idx'] = cnt
                                cnt += 1
                                possible_contours.append(d)

                        temp_result = np.zeros((height, width, channel), dtype = np.uint8)

                        for d in possible_contours:
                            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
                            

                        #Select Candidates by Arrangement of Contours
                        MAX_DIAG_MULTIPLYER = 5
                        MAX_ANGLE_DIFF = 12.0
                        MAX_AREA_DIFF = 0.5
                        MAX_WIDTH_DIFF = 0.8
                        MAX_HEIGHT_DIFF = 0.2
                        MIN_N_MATCHED = 3

                        def find_chars(contour_list):
                            matched_result_idx = []
                            
                            for d1 in contour_list:
                                matched_contours_idx = []
                                for d2 in contour_list:
                                    if d1['idx'] == d2['idx']:
                                        continue
                                        
                                    dx = abs(d1['cx'] - d2['cx'])
                                    dy = abs(d1['cy'] - d2['cy'])
                                    
                                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                                    
                                    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                                    if dx == 0:
                                        angle_diff = 90
                                    else:
                                        angle_diff = np.degrees(np.arctan(dy / dx))
                                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                                    height_diff = abs(d1['h'] - d2['h']) / d1['h']
                                    
                                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                                        matched_contours_idx.append(d2['idx'])
                                        
                                matched_contours_idx.append(d1['idx'])
                                
                                if len(matched_contours_idx) < MIN_N_MATCHED:
                                    continue
                                    
                                matched_result_idx.append(matched_contours_idx)
                                
                                unmatched_contour_idx = []
                                for d4 in contour_list:
                                    if d4['idx'] not in matched_contours_idx:
                                        unmatched_contour_idx.append(d4['idx'])
                                
                                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
                                
                                recursive_contour_list = find_chars(unmatched_contour)
                                
                                for idx in recursive_contour_list:
                                    matched_result_idx.append(idx)
                                    
                                break
                                
                            return matched_result_idx

                        result_idx = find_chars(possible_contours)

                        matched_result = []
                        for idx_list in result_idx:
                            matched_result.append(np.take(possible_contours, idx_list))
                            
                        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

                        for r in matched_result:
                            for d in r:
                                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)

                        #이미지 회전
                        PLATE_WIDTH_PADDING = 1.3 # 1.3
                        PLATE_HEIGHT_PADDING = 1.5 # 1.5
                        MIN_PLATE_RATIO = 3
                        MAX_PLATE_RATIO = 10

                        plate_imgs = []
                        plate_infos = []

                        for i, matched_chars in enumerate(matched_result):
                            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

                            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
                            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
                            
                            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
                            
                            sum_height = 0
                            for d in sorted_chars:
                                sum_height += d['h']

                            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
                            
                            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
                            triangle_hypotenus = np.linalg.norm(
                                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
                            )
                            
                            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
                            
                            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
                            
                            img_rotated = cv2.warpAffine(img_blur_thresh, M=rotation_matrix, dsize=(width, height))
                            
                            img_cropped = cv2.getRectSubPix(
                                img_rotated, 
                                patchSize=(int(plate_width), int(plate_height)), 
                                center=(int(plate_cx), int(plate_cy))
                            )
                            
                            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                                continue
                            
                            plate_imgs.append(img_cropped)
                            plate_infos.append({
                                'x': int(plate_cx - plate_width / 2),
                                'y': int(plate_cy - plate_height / 2),
                                'w': int(plate_width),
                                'h': int(plate_height)
                            })
                            

                        # 글자 찾기
                        longest_idx, longest_text = -1, 0
                        plate_chars = []

                        for i, plate_img in enumerate(plate_imgs):
                            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
                            _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            
                            # find contours again (same as above)
                            contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
                            
                            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
                            plate_max_x, plate_max_y = 0, 0

                            for contour in contours:
                                x, y, w, h = cv2.boundingRect(contour)
                                
                                area = w * h
                                ratio = w / h

                                if area > MIN_AREA \
                                and w > MIN_WIDTH and h > MIN_HEIGHT \
                                and MIN_RATIO < ratio < MAX_RATIO:
                                    if x < plate_min_x:
                                        plate_min_x = x
                                    if y < plate_min_y:
                                        plate_min_y = y
                                    if x + w > plate_max_x:
                                        plate_max_x = x + w
                                    if y + h > plate_max_y:
                                        plate_max_y = y + h
                                        
                            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
                            
                            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
                            _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
                            

                            chars = pytesseract.image_to_string(img_result, lang='kornum+kor', config='--psm 6 preserve_interword_spaces')
                            
                            result_chars = ''
                            has_digit = False
                            for c in chars:
                                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                                    if c.isdigit():
                                        has_digit = True
                                    result_chars += c
                            plate_chars.append(result_chars)

                            if has_digit and len(result_chars) > longest_text:
                                longest_idx = i
#############################################################################################################
                                                    #최종 확인 부문
#############################################################################################################
                        try :
                            info = plate_infos[longest_idx]
                            chars = plate_chars[longest_idx]
                            if ord('가') <= ord(chars[2]) <= ord('힣') and len(chars) == 7:
                                print(chars)
                                excel(chars)
                            elif ord('가') <= ord(chars[3]) <= ord('힣') and len(chars) == 8:
                                print(chars)
                                excel(chars)
                            else :
                                print('인식이 되지 않았습니다 재진입하십시오')

                            img_out = img_ori.copy()

                            cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

                            #cv2.imwrite(chars + '.jpg', img_out) 결과 사진 저장

                            plt.figure(figsize=(12, 10))
                            plt.imshow(img_out)
                            #
                            #          엑셀에 번호판 내용 추가 및 저장
                            #
                        except :
                            print('')
                            #excel()
################################################################################################################
#                                           파일 저장 종료
################################################################################################################                       

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if len(det) != 0:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def excel(chars):
    df = pd.read_excel('HBOS3/plates.xlsx', sheet_name='Sheet1')

    df.loc[len(df)] = [chars]
    # Excel 파일로 저장
    df.to_excel('HBOS3/plates.xlsx', index=False, sheet_name='Sheet1')

    df.info

def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


