import argparse
import socket
import subprocess
import traceback
import random
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
from pathlib import Path
import glob
import numpy as np
import base64
import json
from threading import Lock

import yaml

from networking import *
from jtop import jtop, JtopException
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImagesAndLabels
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, colorstr, box_iou, xywh2xyxy
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync, torch_distributed_zero_first
import torch


m = Lock()

#########################################################################################################
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


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
#########################################################################################################


img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class RootWidget():
    def __init__(self, opt):
        self.is_connect = False
        self.host = None
        self.port = 8009
        self.imgID = 0
        self.inx = 0
        self.strinx = "000000000009.jpg"
        self.update = None
        self.update_pre = None
        self.start = None
        self.det = None
        self.im0 = {}
        self.opt = opt
        self.model = "yolov5s.pt"
        self.user = "c8f6d9ad8d1cdb13"
        self.services = {}
        print(self.user)
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.colors = Colors()
        self.alc_server = None
        self.dag = {
            "device-post-process": True
        }
        self.crt = []
        self.vol = []
        self.to_be_delte = []
        self.gpu_hist = []
        self.detected_result = []
        self.fps = 40
        self.avg_pow = 0
        self.avg_pow_hist = []
        self.edge_fps = 20

        self.real_fps = None
        self.adaptiveFPS = False

        self.images = []  # image files

        self.root_path = "datasets/images/val2017"

        filter = ['000000528399', '000000025593', '000000041488', '000000042888', '000000049091', '000000058636', '000000064574', '000000098497', '000000101022', '000000121153', '000000127135', '000000173183', '000000176701', '000000198915', '000000200152', '000000226111', '000000228771', '000000240767', '000000260657', '000000261796', '000000267946', '000000268996', '000000270386', '000000278006', '000000308391', '000000310622', '000000312549', '000000320706', '000000330554', '000000344611', '000000370999', '000000374727', '000000382734', '000000402096', '000000404601', '000000447789', '000000458790', '000000461275', '000000476491', '000000477118', '000000481404', '000000502910', '000000514540', '000000528977', '000000536343', '000000542073', '000000550939', '000000556498', '000000560371']
        p = Path(self.root_path)  # os-agnostic
        if p.is_dir():  # dir
            images = glob.glob(str(p / '**' / '*.jpg'), recursive=True)
            for im in images:
                # print(im)
                if im.__contains__("detected"):
                    continue
                if im[-16:-4] in filter:
                    continue
                self.images.append(im[:-4])

        print(self.root_path)
        self.time_l = []
        self.time_r = []

        self.send_controller = None

        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.confusion_matrix = ConfusionMatrix(nc=80)
        self.imgsz = 640
        self.create_yolo_service(self.device, self.imgsz, self.half, weights=self.model)
        self.get_labels(imgsz=640, stride=32)

        self.local = False
        self.remote = False

        self.gpu_hist = []
        self.crt_hist = []
        self.vol_hist = []
        self.pow_hist = []
        self.time_hist = []
        self.remote_hist = []

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

    def connect(self):
        self.host = "192.168.1.235"
        self.port = 8001
        if not self.is_connect:
            try:
                self.s = socket.socket()
                self.s.connect((self.host, self.port))
                self.is_connect = True
            except:
                print(traceback.format_exc())
                return

            send_msg(self.s, json.dumps({"content": "handoff", "user": self.user}).encode("utf-8"))

            data = recv_msg(self.s)
            info = json.loads(str(data.decode('utf-8')))
            print("msg:", info)
            if "service" in info and info["service"] == "ready":
                self.start = time.time()
                self.imgsz = info["imgsz"]
                # break

            t2 = threading.Thread(target=self.batter_monitor, args=())
            t2.start()

            self.run()

        else:
            # self.send_controller.cancel()
            self.s.close()
            self.is_connect = False

    def batter_monitor(self):
        with jtop() as jetson:
            # jetson.ok() will provide the proper update frequency
            start = time.time()
            while jetson.ok():
                # GPU
                self.gpu_hist.append(jetson.gpu["val"])
                crt = subprocess.check_output("echo 123456 | sudo cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_current0_input", shell=True)
                self.crt_hist.append(int(crt))
                vol = subprocess.check_output(
                    "echo 123456 | sudo cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_voltage0_input",
                    shell=True)
                self.vol_hist.append(int(vol))
                pow = subprocess.check_output(
                    "echo 123456 | sudo cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input",
                    shell=True)
                self.pow_hist.append(int(pow))
                self.time_hist.append(round(time.time() - start, 2))
                if len(self.time_r) >= 5:
                    self.remote_hist.append(round(np.average(self.time_r[-5:]), 2))
                else:
                    self.remote_hist.append(0)

    def run(self):
        for i in range(2000):
            start_t = time.time()
            items = [
                ("local", self.fps - self.edge_fps),
                ("remote", self.edge_fps)
            ]

            with ThreadPoolExecutor(2) as executor:
                results = executor.map(self.process, items)

            time_1, time_2 = results
            self.time_l.append(time_1)
            self.time_r.append(time_2)

            while time.time() - start_t <= 10:
                pass

            print(
                f"\t video segment {i} (partition {self.fps - self.edge_fps}/{self.edge_fps} #{i}) finished in {round(time.time() - start_t, 4)}/{time_1, time_2},pow={round(np.average(self.pow_hist), 4)}")

            # if i > 0 and (i+1) % 6 == 0:
            self.avg_pow = round(np.average(self.pow_hist), 4)
            self.avg_pow_hist.append(self.avg_pow)
            if i > 0 and i % 10 == 0:
                print("pow_hist=", self.avg_pow_hist)
            self.pow_hist = []

    def process(self, items):
        p_type, fps = items
        if p_type == "local":
            start_t = time.time()
            iid = random.sample(self.images, fps)
            for img_file in iid:
                task = img_file[-12:] + ".jpg"
                try:
                    im0s, img, targets, paths, shapes = self.dataset.__getitem__(task, img_file + ".jpg",
                                                                                 self.imgsz)
                    detect(img, targets, self.device, self.half, self.services[self.model]["model"])
                except:
                    print(traceback.format_exc())
            return round(time.time() - start_t, 4)
        else:
            start_t = time.time()
            if fps > 0:
                self.send_data(fps)
            return round(time.time() - start_t, 4)

    def send_data(self, fps):
        energy_info = {}
        try:
            """
                wait for start_to_send command
            """
            while True:
                data = recv_msg(self.s)
                info = json.loads(str(data.decode('utf-8')))
                if "start_to_send" in info and info["start_to_send"] is True:
                    # print(info)
                    self.edge_fps = info["fps"]
                    self.imgsz = info["imgsz"]
                    self.alc_server = info["server"]
                    break
            """
                send image
            """

            data = []
            ori_file = []

            iid = random.sample(self.images, fps)
            for img_file in iid:
                im0s = cv2.imread(img_file + ".jpg")
                img = cv2.resize(im0s, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
                data.append(base64.b64encode(cv2.imencode('.jpg', img)[1]).decode("utf-8"))
                ori_file.append(img_file)
            if len(self.time_r) > 0:
                msg = {"data": data, "avg_pow": self.avg_pow, "user": self.user, "ori_file": ori_file, "dag": self.dag,
                       "remote_r": self.time_r[-1]}
            else:
                msg = {"data": data, "avg_pow": self.avg_pow, "user": self.user, "ori_file": ori_file, "dag": self.dag}
            encoding = json.dumps(msg).encode("utf-8")
            send_msg(self.s, encoding)

            while True:
                try:
                    data = recv_msg(self.s)
                    info = json.loads(str(data.decode('utf-8')))
                    # print(info)
                    if "status" in info:
                        pass
                    elif "results" in info:
                        break
                except:
                    print(traceback.format_exc())
            start = time.time()
        except:
            print(traceback.format_exc())


@torch.no_grad()
def detect(img, targets,  device, half, model):

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    opt = parser.parse_args()

    r = RootWidget(opt)
    r.connect()
