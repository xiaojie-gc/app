import base64
import os
import json
from threading import Lock
from _thread import *
import socket
import numpy as np
import traceback
import argparse
from pathlib import Path
import cv2
import torch
import yaml

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImagesAndLabels
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, colorstr, box_iou, xywh2xyxy
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync, torch_distributed_zero_first
from networking import *


m = Lock()

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
         'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
         'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

colors = Colors()


class Server:
    def __init__(self, opt):
        self.confusion_matrix = None
        self.host = None
        self.port = None
        self.s = None
        self.ul_channel = []
        self.dl_channel = {}
        self.net = None
        self.pose = None
        self.opt = opt
        self.processed = {}
        self.inx = 0
        self.requests = {}
        self.start = {}
        self.gpu_utilization = []
        self.monitor_on = False
        self.services = {}

        self.device = None
        self.iouv = None
        self.niou  = None
        self.half = None
        self.imgsz = 640

        self.dataset = None

        self.stats = {}
        self.number = {}

        self.segment_time = {}
        self.total_IDLE_time = 0

        self.initial_cfg = {
            # "9fd3b96565dbebe8": 0.2,
            # "204dfa87f106ca64": 0.8
        }
        self.inference = {}

        self.mAP = {
            "yolov5s.pt": {
                "256": [],
                "320": [],
                "416": [],
                "640": []
            },
            "yolov5x.pt": {
                "256": [],
                "320": [],
                "416": [],
                "640": []
            }
        }

    def get_labels(self, imgsz=640, stride=32, task='val', cfg='data/coco128.yaml'):
        with open(cfg) as f:
            data = yaml.safe_load(f)
        # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
        with torch_distributed_zero_first(-1):
            self.dataset = LoadImagesAndLabels(data[task], imgsz, 1,
                                               augment=False,  # augment images
                                               hyp=None,  # augmentation hyperparameters
                                               rect=True,  # rectangular training
                                               cache_images=False,
                                               single_cls=False,
                                               stride=int(stride),
                                               pad=0.5,
                                               image_weights=False,
                                               prefix=colorstr(f'{task}: '))

    def connect_to_urb(self, host='192.168.1.235', port=8000):
        try:
            print("connecting to urb {} ...".format(host))
            urb_channel = socket.socket()
            urb_channel.connect((host, port))
            start_new_thread(self.down_link_handler, (urb_channel, host))
            start_new_thread(self.update_mAP, (urb_channel, ))
            while True:
                try:
                    data = recv_msg(urb_channel)
                    info = json.loads(str(data.decode('utf-8')))

                    task = str(self.inx).zfill(8) + "_" + info["user"]

                    with open("data/images/" + task + ".jpg", 'wb') as file:
                        file.write(base64.b64decode(info["data"]))

                    self.inx += 1

                    if info["user"] not in self.requests:
                        self.requests[info["user"]] = []
                        self.segment_time[info["user"]] = 0

                    m.acquire()
                    if info["user"] not in self.inference:
                        self.inference[info["user"]] = []
                    m.release()

                    if "server" in info:
                        self.initial_cfg[info["user"]] = info["server"]

                    m.acquire()
                    self.requests[info["user"]].append(
                        {"img_file": task, "ori_file": info["ori_file"], "dag": info["dag"], "user": info["user"], "segment": info["segment"],
                              "imgsz": info["imgsz"], "model": info["model"], "total": info["total"]})

                    m.release()
                except error:
                    print(traceback.format_exc())
                    print("urb", host, "disconnected")
        except (RuntimeError, TypeError, NameError):
            urb_channel.close()
            print(traceback.format_exc())

    # 7b0f5c1dbf33e4b2  70
    # db846572170927ce  30

    def down_link_handler(self, urb_channel, urb_addr):

        if len(self.services) == 0:
            self.device = select_device(opt.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
            self.niou = self.iouv.numel()
            self.confusion_matrix = ConfusionMatrix(nc=80)
            self.imgsz = 640
            self.create_yolo_service(self.device, self.imgsz, self.half, weights='yolov5x.pt')
            self.create_yolo_service(self.device, self.imgsz, self.half, weights='yolov5s.pt')
            self.create_yolo_service(self.device, self.imgsz, self.half, weights='yolov5l.pt')
            self.create_yolo_service(self.device, self.imgsz, self.half, weights='yolov5m.pt')
            self.get_labels(imgsz=640, stride=32)

            print("\t service&label ready {}".format(['yolov5s.pt', 'yolov5x.pt', 'yolov5l.pt', 'yolov5m.pt']))
            send_msg(urb_channel, json.dumps({"service": "ready", "model": ['yolov5s.pt', 'yolov5x.pt', 'yolov5l.pt', 'yolov5m.pt']}).encode("utf-8"))

        while True:
            if len(self.initial_cfg.items()) == 1:
                break

        time_window = 0.25
        initialized = False
        while True:
            try:
                total_used = 0
                for user, alloc_time in self.initial_cfg.items():
                    total_used += alloc_time
                idle = 1 - total_used

                if idle > 0:
                    print("GPU IDLE {} = {}".format(user, round(idle * time_window, 4)))
                    time.sleep(idle * time_window)
                    for user1, alloc_time1 in self.initial_cfg.items():
                        if user1 in self.requests and len(self.requests[user1]) != 0:
                            self.segment_time[user1] += idle * time_window
                            print("+ {}:{}".format(user1, round(idle * time_window, 4)))
                for user, alloc_time in self.initial_cfg.items():
                    if user in self.requests and len(self.requests[user]) == 0 or user not in self.requests:
                        print("IDLE because of {} = {}".format(user, round(self.initial_cfg[user] * time_window, 4)))
                        time.sleep(self.initial_cfg[user] * time_window)
                        for user1, alloc_time1 in self.initial_cfg.items():
                            if user1 != user and user1 in self.requests and len(self.requests[user1]) != 0:
                                self.segment_time[user1] += self.initial_cfg[user] * time_window
                                print("+ {}:{}".format(user1, round(self.initial_cfg[user] * time_window, 4)))
                        continue

                    if not initialized:
                        last_busy_time = time.time()
                        initialized = True

                    processed = 0
                    i_time = 0
                    while len(self.requests[user]) > 0:
                        self.total_IDLE_time += time.time() - last_busy_time
                        # print("IDLE = {}".format(self.total_IDLE_time))

                        req = self.requests[user][0]

                        stored_path = "data/images/" + req["img_file"] + ".jpg"
                        task = req["ori_file"][-12:] + ".jpg"

                        im0s, img, targets, paths, shapes = self.dataset.__getitem__(task, stored_path,
                                                                                     req["imgsz"])
                        stats = []

                        start = time.time()
                        msg = detect(im0s, img, targets, paths, shapes, self.device, self.half,
                                     self.services[req["model"]]["model"],
                                     req["dag"], self.iouv, self.niou, self.confusion_matrix,
                                     stats)
                        msg["inference"] = round(time.time() - start, 4)
                        self.inference[req["user"]].append(msg["inference"])
                        self.segment_time[req["user"]] += msg["inference"]
                        i_time += msg["inference"]

                        if req["user"] not in self.stats:
                            self.stats[req["user"]] = []
                            self.number[req["user"]] = 0

                        self.number[req["user"]] += 1

                        for item in stats:
                            self.stats[req["user"]].append(item)

                        map50 = None
                        inference = None
                        segment = None
                        if self.number[req["user"]] >= req["total"]:
                            start_r = time.time()
                            ap_list = []
                            if len(self.stats[req["user"]]) > 0:
                                records = [np.concatenate(x, 0) for x in zip(*self.stats[req["user"]])]  # to numpy
                                if len(records) and records[0].any():
                                    _, _, ap, _, _ = ap_per_class(*records, plot=False, save_dir=None,
                                                                  names=names)
                                    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                                    ap_list = np.concatenate([ap_list, ap])
                            self.stats[req["user"]] = []
                            self.number[req["user"]] = 0
                            map50 = round(ap_list.mean(), 4)
                            inference = np.average(self.inference[req["user"]])
                            segment = self.segment_time[req["user"]]
                            self.inference[req["user"]] = []
                            self.segment_time[req["user"]] = 0
                            print("+ eva", round(time.time() - start_r, 4))

                        msg["seg_time"] = segment
                        msg["inference"] = inference
                        msg["mAP50"] = map50
                        msg["model"] = req["model"]
                        msg["imgsz"] = req["imgsz"]
                        msg["FPS"] = 0
                        msg["ori_file"] = req["ori_file"]
                        msg["user"] = req["user"]
                        msg["total_IDLE_time"] = self.total_IDLE_time
                        msg["segment"] = req["segment"]
                        send_msg(urb_channel, json.dumps(msg).encode("utf-8"))
                        self.requests[user].remove(req)
                        processed += 1
                        try:
                            os.remove(stored_path)
                        except:
                            pass
                        last_busy_time = time.time()
                        if i_time >= self.initial_cfg[user] * time_window:
                            break
                    print("{}[{}%]: processed {}, remaining {}, latency {}"
                          .format(user, round(self.initial_cfg[user] * 100),
                                  processed,
                                  len(self.requests[user]),
                                  round(np.average(self.inference[req["user"]]), 4)))
            except:
                print(traceback.format_exc())
                print("urb", urb_addr, "disconnected")

    def update_mAP(self, urb_channel):
        start = time.time()
        while True:
            if time.time() - start < 5:
                time.sleep(5)
            start = time.time()
            try:
                mAP = {}
                for model, hist in self.mAP.items():
                    mAP[model] = {}
                    for resolution, ap in hist.items():
                        if len(ap) > 0:
                            mAP[model][resolution] = np.average(ap)
                        else:
                            mAP[model][resolution] = 0
                msg = {"mAP": mAP}
                send_msg(urb_channel, json.dumps(msg).encode("utf-8"))
            except:
                print(traceback.format_exc())

    def create_yolo_service(self, device, imgsz, half, weights='yolov5x.pt'):
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        check_img_size(imgsz, s=stride)
        if half:
            model.half()  # to FP16

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        self.services[weights] = {
            "model": model,
            "stride": stride,
            "name": weights
        }


def process_batch(predictions, labels, iouv):
    # Evaluate 1 batch of predictions
    correct = torch.zeros(predictions.shape[0], len(iouv), dtype=torch.bool, device=iouv.device)
    detected = []  # label indices
    tcls, pcls = labels[:, 0], predictions[:, 5]
    nl = labels.shape[0]  # number of labels
    for cls in torch.unique(tcls):
        ti = (cls == tcls).nonzero().view(-1)  # label indices
        pi = (cls == pcls).nonzero().view(-1)  # prediction indices
        if pi.shape[0]:  # find detections
            ious, i = box_iou(predictions[pi, 0:4], labels[ti, 1:5]).max(1)  # best ious, indices
            detected_set = set()
            for j in (ious > iouv[0]).nonzero():
                d = ti[i[j]]  # detected label
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)  # append detections
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all labels already located in image
                        break
    return correct


@torch.no_grad()
def detect(im0s, img, targets, paths, shapes,  device, half, model, dag, iouv, niou, confusion_matrix, stats):

    # im0s = cv2.imread(paths[0])  # BGR

    img = img.to(device, non_blocking=True)
    targets = targets.to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    nb, _, height, width = img.shape  # batch size, channels, height, width

    targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)

    # Inference
    out, train_out = model(img, augment=opt.augment)

    # Apply NMS
    out = non_max_suppression(out, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                              max_det=opt.max_det)

    msg = {"data": None, "dag": dag}

    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path, shape = Path(paths[si]), shapes[si][0]
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        predn = pred.clone()
        scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
            confusion_matrix.process_batch(predn, labelsn)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        # Result

        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()

            if dag["device-post-process"] is True:
                msg = {"data": pred.cpu().numpy().tolist(),  "dag": dag}
            else:
                for *xyxy, conf, cls in reversed(pred):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=colors(c, True), line_thickness=3)
                msg = {"data": base64.b64encode(cv2.imencode('.jpg', im0s)[1]).decode("utf-8"),  "dag": dag}

    return msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')

    parser.add_argument('--port', type=int, default=8009, help='maximum number of detections per image')

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    server = Server(opt)
    server.connect_to_urb(host='192.168.1.235', port=opt.port)

