import json


class AudioFeatureTool(object):

    def __init__(self):
        self.audio_dict = None

    def get_all_from_origin_file(self, audio_file_path):
        video_file = open(audio_file_path, 'r')
        for line in video_file.readlines():
            item = json.loads(line)
            self.audio_dict[item["item_id"]] = item["video_feature_dim_128"]
        video_file.close()
        return self.audio_dict

    def get(self, item_id):
        return self.audio_dict.get(item_id, list())

