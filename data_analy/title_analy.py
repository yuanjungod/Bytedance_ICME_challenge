import json


class TitleAnalyTool(object):

    def __init__(self, title_file_path, debug=False):
        # /Volumes/Seagate Expansion Drive/byte/track2/track2_title.txt
        title_file = open(title_file_path)

        self.origin_feature_dict = dict()
        self.words_count_dict = dict()
        count = 0
        line = title_file.readline()
        while line:
            count += 1
            if count > 10000 and debug:
                break
            if count % 10000 == 0:
                print(count)
            item = json.loads(line)
            for key, values in item["title_features"].items():
                if key not in self.words_count_dict:
                    self.words_count_dict[key] = 0
                self.words_count_dict[key] += values
            self.origin_feature_dict[item["item_id"]] = item["title_features"]
            line = title_file.readline()
        title_file.close()



