f = open("/Users/quantum/Downloads/final_track2_train.txt", "r")

min_time = 9999999999999
during_time = 0
result_dict = dict()
for line in f.readlines():
    line_list = line.split("\t")
    min_time = min(min_time, int(line_list[10]))
    during_time = max(during_time, int(line_list[11]))
    uid = int(line_list[0])
    if int(uid) not in result_dict:
        result_dict[int(uid)] = list()
    result_dict[int(uid)].append(line)
f.close()


min_num = 0
max_num = 0
for i in result_dict:
    min_num = min(len(result_dict[i]), min_num)
    max_num = max(len(result_dict[i]), max_num)
