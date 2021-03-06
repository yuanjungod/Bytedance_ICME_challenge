from model_zoo.din_xdeep_bert import DeepFM
import torch
import os
import json
import pandas

result_dict = {"uid": [], "item_id": [], "finish_probability": [], "like_probability": []}
# cal_result = dict()
deep_fm = DeepFM(10, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, 500000], 128, 128,
                 embedding_size=128, learning_rate=0.003, use_bert=True, num_attention_heads=8,
                 batch_size=256, deep_layers_activation='sigmoid')
result_list = list()


def submit():

    # model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/finish/20190223/byte_25000.model'
    # model_path = '/Volumes/Seagate Expansion Drive/byte/track2/models/20190304/byte_295000.model'
    model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/20190304/byte_295000.model'

    # deep_fm.load_state_dict(torch.load(model_path, map_location='cpu'))
    deep_fm.load_state_dict(torch.load(model_path))
    deep_fm.cuda(0)
    submit_path_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/submit_jsons/"
    # submit_path_dir = "/Volumes/Seagate Expansion Drive/byte/track2/submit_jsons"
    submit_files = [os.path.join(submit_path_dir, i) for i in os.listdir(submit_path_dir)]

    for file in submit_files:
        print(file)
        fp = open(file, "r")
        result = json.load(fp)
        fp.close()

        # print("audio", result["audio"][0], type(result["audio"][0]))
        # print("video", result["video"][0], type(result["video"][0]))

        step = 512
        for i in range(0, len(result["index"]), step):
            like_preb, finish_preb = deep_fm.predict_proba(
                result["index"][i: i+step], result["value"][i: i+step], result["video"][i: i+step],
                result["audio"][i: i+step], result["title"][i: i+step], result["title_value"][i: i+step])

            like_preb = like_preb[:, 1]
            finish_preb = finish_preb[:, 1]
            print(like_preb.shape, finish_preb.shape)
            for j in range(len(result["index"][i: i+step])):
                uid = result["index"][i+j][0]
                item_id = result["item_id"][i+j]
                result_list.append([uid, item_id, like_preb[j], finish_preb[j]])


submit()


for item in result_list:
    result_dict["uid"].append(item[0]-1)
    result_dict["item_id"].append(item[1])
    result_dict["like_probability"].append(item[2])
    result_dict["finish_probability"].append(item[3])

pandas.DataFrame(result_dict).to_csv("result.csv", index=False)



