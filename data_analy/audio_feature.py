import json


class AudioFeatureTool(object):

    def __init__(self):
        self.audio_dict = dict()

    def get_all_from_origin_file(self, audio_file_path):
        audio_file = open(audio_file_path, 'r')
        line = audio_file.readline()
        count = 0
        while line:
            count += 1
            if count % 1000000 == 0:
                print("audio", count, len(self.audio_dict))
            # print(line)
            item = json.loads(line)
            self.audio_dict[item["item_id"]] = json.dumps(item["audio_feature_128_dim"])
            line = audio_file.readline()
        audio_file.close()
        return self.audio_dict

    def get(self, item_id):
        return self.audio_dict.get(item_id, list())

