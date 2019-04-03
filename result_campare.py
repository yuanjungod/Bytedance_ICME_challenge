import pandas as pd


def pdtodict(pd_path):
    pd1 = pd.read_csv(pd_path)
    pd1_dict = pd1.to_dict()
    print(list(pd1_dict.keys()))
    dict1 = dict()
    uid_list = pd1_dict["uid"]
    item_id_list = pd1_dict["item_id"]
    finish_probability_list = pd1_dict["finish_probability"]
    like_probability_list = pd1_dict["like_probability"]

    for i in range(len(uid_list)):
        if uid_list[i] not in dict1:
            dict1[uid_list[i]] = dict()
        if item_id_list[i] not in dict1[uid_list[i]]:
            dict1[uid_list[i]][item_id_list[i]] = [finish_probability_list[i], like_probability_list[i]]
        else:
            # print("*"*30, uid_list[i], item_id_list[i])
            # print(dict1[uid_list[i]][item_id_list[i]])
            # print([finish_probability_list[i], like_probability_list[i]])
            dict1[uid_list[i]][item_id_list[i]] = [finish_probability_list[i], like_probability_list[i]]
    return dict1


pd_path1 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190316/1/result.csv"
dict1 = pdtodict(pd_path1)

pd_path2 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190316/result.csv"
dict2 = pdtodict(pd_path2)

pd_path3 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190318/result.csv"
dict3 = pdtodict(pd_path3)

pd_path4 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190320/result.csv"
dict4 = pdtodict(pd_path4)

pd_path5 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190321/result.csv"
dict5 = pdtodict(pd_path5)

pd_path6 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190322/result.csv"
dict6 = pdtodict(pd_path6)

pd_path7 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190323/result.csv"
dict7 = pdtodict(pd_path7)

pd_path8 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190324/result.csv"
dict8 = pdtodict(pd_path8)

pd_path9 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190325/result.csv"
dict9 = pdtodict(pd_path9)

pd_path10 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190326/result.csv"
dict10 = pdtodict(pd_path10)

pd_path11 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190331/1/result.csv"
dict11 = pdtodict(pd_path11)

pd_path12 = "/Users/quantum/code/Bytedance_ICME_challenge/result/20190331/result.csv"
dict12 = pdtodict(pd_path12)

result_dict = {"uid": [], "item_id": [], "finish_probability": [], "like_probability": []}

result_list = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, dict12]


def merge():
    pd_finish1 = pd.read_csv("./result/20190324/result_finish.csv")
    uid_list = pd_finish1["uid"]
    item_id_list = pd_finish1["item_id"]

    for i in range(len(uid_list)):
        uid = uid_list[i]
        item_id = item_id_list[i]
        result_dict['uid'].append(uid)
        result_dict['item_id'].append(item_id)
        result_dict['finish_probability'].append(sum([i[uid][item_id][0] for i in result_list])/len(result_list))
        result_dict['like_probability'].append(sum([i[uid][item_id][1] for i in result_list])/len(result_list))


def show(uid, item_id):
    # print(dict1[uid][item_id])
    # print(dict2[uid][item_id])
    print("finish", dict3[uid][item_id])
    print("like", dict4[uid][item_id])


merge()

