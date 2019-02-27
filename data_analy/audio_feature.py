import json


class AudioFeatureTool(object):

    def __init__(self):
        self.audio_dict = None

    def get_all_from_origin_file(self, audio_file_path):
        video_file = open(audio_file_path, 'r')
        line = video_file.readline()
        count = 0
        while line:
            count += 1
            if count % 1000000 == 0:
                print("audio", count, len(self.audio_dict))
            # print(line)
            item = json.loads(line)
            self.audio_dict[item["item_id"]] = item["audio_feature_128_dim"]
        video_file.close()
        return self.audio_dict

    def get(self, item_id):
        return self.audio_dict.get(item_id, list())

