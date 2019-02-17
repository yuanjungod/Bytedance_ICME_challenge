import json

title_file = open("/Volumes/Seagate Expansion Drive/byte/track2/track2_title.txt")

title_feature_dict = dict()
for i in title_file.readlines():
    item = json.loads(i)
    for key, value in item['title_features'].items():
        if key not in title_feature_dict:
            title_feature_dict[key] = value
        title_feature_dict[key] += value

print(len(title_feature_dict))
