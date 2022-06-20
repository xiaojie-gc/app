import math
import time
from pathlib import Path
import base64
import os
import random
import json
from threading import Lock
from _thread import *
import socket
import cv2
import numpy as np
import traceback
import argparse
from networking import *

m = Lock()  

m_task = Lock()


class Controller:
    def __init__(self, opt):
        self.ue_ul_channel = []
        self.ue_dl_channel = {}
        self.requests = {}
        self.server_channel = {}
#        self.alg = opt.alg
        self.server_list = []
        self.ue_server_mapping = {}
        self.mAP = {}
        self.resolution = [256, 320, 416, 640]
        self.user_info = {}
        self.stats = {}
        self.tasks = {}
        self.send_enable = {}
        self.networking = {}
        self.inference = {}
        self.initial_cfg = {
            "c8f6d9ad8d1cdb13":  {                   # Id = 1 002
                "model": "yolov5s.pt",
                "imgsz": 512,
                "fps": 20,
                "remote_r": None,
                "deadline": 10,
                "total_fps": 40,
                "avg_pow": 0,
                "server": 0.5,
                "segment": 0,
                "pre_u_segment": None
            },
            "204dfa87f106ca64": {                   # Id = 0  005
                "model": "yolov5x.pt",
                "imgsz": 512,
                "fps": 15,
                "remote_r": None,
                "deadline": 10,
                "total_fps": 30,
                "avg_pow": 0,
                "server": 0.5,
                "segment": 0,
                "pre_u_segment": None
            }
        }
        self.confusion_matrix = None
        self.net = None
        self.pose = None
        self.opt = opt
        self.processed = {}
        self.inx = 0
        self.requests = {}
        self.start = {}
        self.services = {}
        self.device = None
        self.iouv = None
        self.niou = None
        self.half = None
        self.imgsz = 640
        self.dataset = None
        self.stats = {}
        self.number = {}
        self.total_IDLE_time = 0
        self.ue_server_mapping = {}
        self.mAP = {}
        self.resolution = [256, 320, 416, 640]
        self.user_info = {}
        self.stats = {}
        self.tasks = {}
        self.send_enable = {}
        self.server_list = []
        self.segment_time = {}
        self.ue_ul_channel = []
        self.ue_dl_channel = {}
        self.requests = {}
        self.server_channel = {}
#        self.alg = opt.alg

        self.current = {}
        self.vol = {}

        start_new_thread(self.ue_to_controller, (8001,))

    """
           Communicate with servers
    """

    def server_to_controller(self, port=8000):
        self.dispatcher_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dispatcher_socket.bind(("", port))
        self.dispatcher_socket.listen(2)
        print("start to listening -> server......")
        start_new_thread(self.update_allocation, ())
        while True:
            try:
                server_channel, server_addr = self.dispatcher_socket.accept()
                print("\t server connected, start to communicate with server [{}]".format(server_addr))
                self.server_channel[server_addr] = server_channel
                self.server_list.append({
                    'host': server_addr,
                    'model': None,
                    'inference': {},
                    "total_IDLE_time": 0
                })
                start_new_thread(self.fetcher, (server_channel, server_addr))
                print(self.server_list)
            except:
                self.dispatcher_socket.close()
                print(traceback.format_exc())
                return

    """
          Get a valid server address
      """

    def get_server(self):
        while len(self.server_list) == 0:
            time.sleep(1)
        return self.server_list[0] if random.randint(0, 100) < 30 else self.server_list[1]


    """
        Communicate with users
    """
    def ue_to_controller(self, port=8001):
        try:
            self.socket_ue = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_ue.bind(("", port))
            self.socket_ue.listen(5)
        except Exception:
            print(traceback.format_exc())

        print("start to listening -> mobile device......")
        while True:
            try:
                ue_channel, addr = self.socket_ue.accept()
                print("\t client {} connected, start to establish down-link channel......".format(addr))
                self.ue_ul_channel.append(ue_channel)
                data = recv_msg(ue_channel)
                info = json.loads(str(data.decode('utf-8')))
                if "content" in info and info["content"] == "handoff":
                    print("\t Hi, {}".format(info["user"]))
                    if info["user"] not in self.ue_dl_channel:
                        """
                            User mAP and task scheduler information
                        """
                        self.user_info[info["user"]] = {
                            "map": [],
                            "total": 0,
                            "task": {},
                            "networking": 0,
                            "inference": 0
                        }

                        for server in self.server_list:
                            self.user_info[info["user"]]["task"][str(server["host"])] = {
                                    "total": 0,
                                    "yolov5s.pt": {
                                        "256": 0,
                                        "320": 0,
                                        "416": 0,
                                        "640": 0
                                    },
                                    "yolov5x.pt": {
                                        "256": 0,
                                        "320": 0,
                                        "416": 0,
                                        "640": 0
                                    }
                                }

                        time.sleep(1)

                        self.segment_time[info["user"]] = []
                        self.requests[info["user"]] = []
                        self.tasks[info["user"]] = []
                        self.networking[info["user"]] = []
                        self.inference[info["user"]] = []

                    self.send_enable[info["user"]] = True
                    self.ue_dl_channel[info["user"]] = ue_channel


                    msg = {"service": "ready", "fps": self.initial_cfg[info["user"]]["fps"],
                               "imgsz": self.initial_cfg[info["user"]]["imgsz"]}
                    send_msg(ue_channel, json.dumps(msg).encode("utf-8"))
                    print("\t down-link channel established for {}".format(info['user']))

                    start_new_thread(self.ue_ofdma_controller, (ue_channel, info['user']))


            except:
                self.socket_ue.close()
                print(traceback.format_exc())
                break

    """
        Get a valid server address
    
    def get_server(self):
        while len(self.server_list) == 0:
            time.sleep(1)
        return self.server_list[0] if random.randint(0, 100) < 30 else self.server_list[1]
    """

    """
        Receive detection results from server and forward those results to mobile device 
    """
    def fetcher(self, server_channel, server_addr):
        while True:
            try:
                data = recv_msg(server_channel)
                info = json.loads(str(data.decode('utf-8')))

                if "service" in info:
                    pass
                elif "mAP" in info:
                    pass
                else:
                    if info["mAP50"] is not None:
                        self.user_info[info["user"]]["map"].append(round(info["mAP50"], 4))

                    if info["inference"] is not None:
                        self.inference[info["user"]].append(round(info["inference"], 4))
                        self.segment_time[info["user"]].append(round(info["seg_time"], 4))

                    self.user_info[info["user"]]["total"] += 1
                    self.tasks[info["user"]][info["segment"]]["results"].append({
                        "ori_file": info["ori_file"],
                        "data": info["data"]
                    })
            except:
                print(traceback.format_exc())
                return

    def empty_task_info(self, user):
        self.inference[user] = []
        self.segment_time[user] = []
        self.user_info[user]["map"] = []
        return {
                "static": {
                    "time": [],
                    "energy": [],
                    "vol": [],
                    "crt": []
                },
                "network": {
                    "time": [],
                    "energy": [],
                    "vol": [],
                    "crt": []
                },
                "encoding": {
                    "time": [],
                    "energy": [],
                    "vol": [],
                    "crt": []
                }
            }

    def update_task_info(self, info, energy_info):
        if np.isnan(info["energy_info"]["static"]["energy"]):
            return energy_info
        if np.isnan(info["energy_info"]["network"]["energy"]):
            return energy_info

        static_ = round(-info["energy_info"]["static"]["energy"] * info["energy_info"]["static"]["time"] *
                        info["energy_info"]["static"]["vol"] * 0.001, 2)
        network_ = round(
            -info["energy_info"]["network"]["energy"] * info["energy_info"]["network"]["time"] *
            info["energy_info"]["network"]["vol"] * 0.001, 2)

        encoding_ = round(
            -info["energy_info"]["encoding"]["energy"] * info["energy_info"]["encoding"]["time"] *
            info["energy_info"]["encoding"]["vol"] * 0.001, 2)

        energy_info["static"]["energy"].append(static_)
        energy_info["network"]["energy"].append(network_)
        energy_info["encoding"]["energy"].append(encoding_)

        energy_info["static"]["time"].append(info["energy_info"]["static"]["time"])
        energy_info["network"]["time"].append(info["energy_info"]["network"]["time"])
        energy_info["encoding"]["time"].append(info["energy_info"]["encoding"]["time"])

        energy_info["static"]["vol"].append(info["energy_info"]["static"]["vol"])
        energy_info["network"]["vol"].append(info["energy_info"]["network"]["vol"])
        energy_info["encoding"]["vol"].append(info["energy_info"]["encoding"]["vol"])

        energy_info["static"]["crt"].append(info["energy_info"]["static"]["energy"])
        energy_info["network"]["crt"].append(info["energy_info"]["network"]["energy"])
        energy_info["encoding"]["crt"].append(info["energy_info"]["encoding"]["energy"])

        return energy_info

    def display_task_info(self, user, segment, energy_info, energy, inference, network, u_map, crt, vol, seg_inference):
        print(
            f"####{segment}################### {user}, {self.initial_cfg[user]['model']}, {self.initial_cfg[user]['imgsz']} ############################")
        x = np.array(
            [np.average(energy_info["static"]["vol"]), np.average(energy_info["encoding"]["vol"]),
             np.average(energy_info["network"]["vol"])])
        x = np.round(x, 4)

        vol.append(round(x[2], 2))

        x = np.array(
            [np.average(energy_info["static"]["crt"]), np.average(energy_info["encoding"]["crt"]),
             np.average(energy_info["network"]["crt"])])
        x = np.round(x, 4)

        crt.append(round(-x[2], 2))

        x = np.array(
            [np.average(energy_info["static"]["energy"]), np.average(energy_info["encoding"]["energy"]),
             np.average(energy_info["network"]["energy"])])
        x = np.round(x, 4)

        energy.append(round(x[2], 2))

        t = np.array(
            [np.average(energy_info["static"]["time"]), np.average(energy_info["encoding"]["time"]),
             np.average(energy_info["network"]["time"])])
        t = np.round(t, 4)

        inference.append(round(np.average(self.inference[user]), 4))
        network.append(t[2])
        u_map.append(round(np.average(self.user_info[user]["map"]), 4))
        seg_inference.append(round(np.average(self.segment_time[user]), 4))


        print("network = ", network)
        print("inference = ", inference)
        print("inference_SEG = ", seg_inference)
        print("energy = ", energy)
        print("crt = ", crt)
        print("vol = ", vol)
        print("map = ", u_map)
        print(f"#####################################################################")
        return energy, inference, network, u_map, crt, vol, seg_inference

    def update_allocation(self):
        rate_parameter = 1 / 180
        event_start = time.time()
        next_t = round(nextTime(rate_parameter), 0)
        total = round(random.uniform(0.3, 0.7), 2)
        print("new resource:", total)
        print("new duration:", next_t)
        for user, values in self.initial_cfg.items():
            self.initial_cfg[user]["server"] = total/2
            print(f"\tnew allocation: {user}={self.initial_cfg[user]['server']}")

        while True:
            if time.time() - event_start >= next_t:
                event_start = time.time()
                next_t = round(nextTime(rate_parameter), 0)
                total = round(random.uniform(0.3, 0.7), 2)
                print("new resource:", total)
                print("new duration:", next_t)
            else:
                pass
            pow_sum = 0
            for user, values in self.initial_cfg.items():
                pow_sum += self.initial_cfg[user]["avg_pow"]
            if pow_sum != 0:
                for user, values in self.initial_cfg.items():
                    self.initial_cfg[user]["server"] = round(total * (self.initial_cfg[user]["avg_pow"] / pow_sum), 2)

    """
        Get image from user and save to request queue
    """
    def ue_ofdma_controller(self, ue_channel, user):
        segment = 0
        try:
            while True:
                send_msg(ue_channel, json.dumps({"start_to_send": True, "fps": self.initial_cfg[user]["fps"],
                                                 "imgsz": self.initial_cfg[user]["imgsz"],
                                                 "server": self.initial_cfg[user]["imgsz"]}).encode("utf-8"))
                self.get_ue_task(ue_channel, user, segment)
                self.initial_cfg[user]["segment"] += 1
                segment += 1
                print(user, ":avg_pow=", self.initial_cfg[user]["avg_pow"], "GPU usage=", self.initial_cfg[user]["server"])
                if self.initial_cfg[user]["remote_r"] is None or self.initial_cfg[user]["segment"] == 0:
                    continue
                if self.initial_cfg[user]["remote_r"] > self.initial_cfg[user]["deadline"]:
                    self.initial_cfg[user]["fps"] = max(1, self.initial_cfg[user]["fps"] - 2)
                else:
                    self.initial_cfg[user]["fps"] = min(self.initial_cfg[user]["total_fps"], self.initial_cfg[user]["fps"] + 2)
        except:
            print(traceback.format_exc())
            self.socket_ue.close()

    def get_ue_task(self, ue_channel, user, segment):
        try:
            self.tasks[user].append({
                "total": 0,
                "results": [],
                "networking": 0,
                "inference": 0
            })
            self.tasks[user][segment]["networking"] = time.time()
            data = recv_msg(ue_channel)
            send_msg(ue_channel, json.dumps({"status": "ok"}).encode("utf-8"))
            self.networking[user].append(time.time() - self.tasks[user][segment]["networking"])
            info = json.loads(str(data.decode('utf-8')))
            self.initial_cfg[user]["avg_pow"] = info["avg_pow"]
            self.insert_tasks(info, segment, self.server_channel[self.server_list[0]["host"]])
            self.tasks[info["user"]][segment]["inference"] = time.time()
            self.get_ue_task_results(ue_channel, info["user"], info["dag"], segment)
        except:
            ue_channel.close()
            self.socket_ue.close()
            print(traceback.format_exc())

    def get_ue_task_results(self, ue_channel, user, dag, segment):
        try:
            while len(self.tasks[user][segment]["results"]) != self.tasks[user][segment]["total"]:
                pass
            msg = {"dag": dag, "data": [], "ori_file": [], "results": True}
            for i in range(self.tasks[user][segment]["total"]):
                msg["data"].append(self.tasks[user][segment]["results"][i]["data"])
                msg["ori_file"].append(self.tasks[user][segment]["results"][i]["ori_file"])
            # self.inference[user].append((time.time() - self.tasks[user][segment]["inference"]))
            send_msg(ue_channel, json.dumps(msg).encode("utf-8"))
        except (RuntimeError, TypeError, NameError):
            print(traceback.format_exc())
            print("client", user, "disconnected")
            ue_channel.close()

    def insert_tasks(self, info, segment, server_channel):
        try:
            if "remote_r" in info:
                self.initial_cfg[info["user"]]["remote_r"] = info["remote_r"]
                #print("remote_r of ", info["user"], "is", info["remote_r"])

            imgsz = self.initial_cfg[info["user"]]["imgsz"]
            model = self.initial_cfg[info["user"]]["model"]
            total = len(info["data"])
            self.tasks[info["user"]][segment]["total"] = total
            for i in range(total):
                task = str(self.inx).zfill(8) + "_" + info["user"]
                # with open("data/images/" + task + ".jpg", 'wb') as file:
                #    file.write(base64.b64decode(info["data"][i]))
                self.inx += 1
                req = {"imgsz": imgsz, "model": model, "user": info["user"], "dag": info["dag"], "total": total,
                       "segment": segment,
                       "data": info["data"][i], "ori_file": info["ori_file"][i],
                       "server": self.initial_cfg[info["user"]]["server"]}
                # m.acquire()
                send_msg(server_channel, json.dumps(req).encode("utf-8"))
                # m.release()
        except (RuntimeError, TypeError, NameError):
            print(traceback.format_exc())


def nextTime(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8009, help='maximum number of detections per image')
    opt = parser.parse_args()
    print(opt)

    try:
        urb = Controller(opt)
        urb.server_to_controller(opt.port)
    except:
        urb.socket_ue.close()
        print('Interrupted')
