import json


class VideoFeature(object):

    def __init__(self, video_feature_path, debug=False):
        video_file = open(video_feature_path, 'r')
        self.video_dict = dict()

        count = 0
        line = video_file.readline()
        while line:
            count += 1
            if count > 10000 and debug:
                break
            if count % 100000 == 0:
                print(count)
            # print(line)
            item = json.loads(line)
            self.video_dict[item["item_id"]] = item["video_feature_dim_128"]
            line = video_file.readline()
        video_file.close()

    def get_video_embedding(self, item_id):
        return self.video_dict[item_id]
