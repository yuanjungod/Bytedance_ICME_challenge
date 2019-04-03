from pymemcache.client.base import Client
from .data_preprocessor import DataPreprocessor


class Memcache(object):

    Title = "title"
    Video = "video"
    Audio = "audio"
    User = "user"

    def __init__(self, host, port):
        self.client = Client((host, port))

    def insert(self, data_tool):
        data_tool.get_origin_train_data()
        for key in data_tool.title_feature_tool.result_dict:
            self.client.set("%s%s" % (self.Title, key), data_tool.title_feature_tool.self.result_dict[key])

        for key in data_tool.video_feature_tool.video_dict:
            self.client.set("%s%s" % (self.Video, key), data_tool.video_feature_tool.video_dict[key])

        for key in data_tool.audio_feature_tool.audio_dict:
            self.client.set("%s%s" % (self.Audio, key), data_tool.audio_feature_tool.audio_dict[key])

        for key in data_tool.audio_feature_tool.audio_dict:
            self.client.set("%s%s" % (self.Audio, key), data_tool.audio_feature_tool.audio_dict[key])

        for i in range(len(data_tool.user_action_list)):
            self.client.set("%s%s" % (self.User, i), data_tool.user_action_list[i])

    def get_title(self, item_id):
        self.client.get("%s%s" % (self.Title, item_id))

    def get_video(self, item_id):
        self.client.get("%s%s" % (self.Video, item_id))

    def get_audio(self, item_id):
        self.client.get("%s%s" % (self.Audio, item_id))

    def get_user(self, index):
        self.client.get("%s%s" % (self.User, index))


if __name__ == "__main__":
    mem_cache = Memcache("local", 11211)
    data_tool = DataPreprocessor()
    mem_cache.insert(data_tool)
