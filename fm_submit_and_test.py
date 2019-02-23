from model_zoo.deep_fm import DeepFM
import torch
import os

task = "like"
deep_fm = DeepFM(9, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20], 128, task)

model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/finish/20190223/byte_25000.model'
deep_fm.load_state_dict(torch.load(model_path))
deep_fm.cuda(0)

submit_path_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/submit_jsons/"

submit_files = [os.path.join(submit_path_dir, i) for i in os.listdir(submit_path_dir)]


