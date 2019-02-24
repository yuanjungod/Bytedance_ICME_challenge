from model_zoo.deep_fm import DeepFM
import torch
import os
import json
import pandas

result_dict = {"uid": [], "item_id": [], "finish_probability": [], "like_probability": []}
cal_result = dict()
deep_fm = DeepFM(9, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20], 128, "like")


def submit(task):
    if task == "like":
        model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/like/20190223/byte_25000.model'
    else:
        model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/finish/20190223/byte_25000.model'

    deep_fm.load_state_dict(torch.load(model_path))
    deep_fm.cuda(0)
    submit_path_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/submit_jsons/"
    submit_files = [os.path.join(submit_path_dir, i) for i in os.listdir(submit_path_dir)]

    for file in submit_files:
        fp = open(file, "r")
        result = json.load(fp)
        fp.close()

        predict_proba = deep_fm.predict_proba(result["index"], result["value"], result["video"], result["title"], result["title_value"])

        for i in range(len(result["index"])):
            uid = result["index"][i][0]
            item_id = result["item_id"][i]
            if uid not in cal_result:
                cal_result[uid] = dict()
            if item_id not in cal_result[uid]:
                cal_result[uid][item_id] = dict()
            if task == "like":
                cal_result[uid][item_id]["like_probability"] = predict_proba[i]
            else:
                cal_result[uid][item_id]["finish_probability"] = predict_proba[i]


submit("like")
submit("finish")
for uid in cal_result:
    for item_id in cal_result[uid]:
        result_dict["uid"].append(uid)
        result_dict["item_id"].append(item_id)
        result_dict["finish_probability"].append(cal_result[uid][item_id]["finish_probability"])
        result_dict["like_probability"].append(cal_result[uid][item_id]["like_probability"])
pandas.DataFrame(result_dict).to_csv("result.csv")



