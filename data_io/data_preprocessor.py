from data_analy.video_feature import *
from data_analy.user_interactive import *
from data_analy.title_analy import *
from data_analy.audio_feature import *
import json
import random

random.seed(941)


class DataPreprocessor(object):

    FIELD_SIZE = 9
    FEATURE_SIZES = [80000, 400, 900000, 500, 5, 90000, 80000, 30, 20]
    MAX_TITLE_SIZE = 30

    def __init__(self, video_db_path=None, user_db_path=None, title_feature_path=None):
        print("init video feature")
        self.video_feature_tool = VideoFeature(video_db_path)
        print("init audio feature")
        self.audio_feature_tool = AudioFeatureTool()
        print("init title feature")
        self.title_feature_tool = TitleAnalyTool(title_feature_path)
        print("init user feature")
        self.user_interactive_tool = UserInteractiveTool(user_db_path)

    def get_train_data(self):
        result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [], 'item_id': [],
                  "video": [], 'feature_sizes': self.FEATURE_SIZES, 'tile_word_size': self.title_feature_tool.MAX_WORD}
        step = 500
        for i in range(0, self.user_interactive_tool.get_max_id("ID"), step):
            print("data loading")
            users = self.user_interactive_tool.get(i, i+step)
            for user in users:
                user_action, item_id, like, finish = user
                result['item_id'].append(item_id)
                result['like'].append(like)
                result['finish'].append(finish)
                result['index'].append(user_action)
                result['value'].append([1 for _ in user_action])
                title_list = list(self.title_feature_tool.get(item_id))
                result['title'].append([title_list[i] if i < len(title_list) else 0 for i in range(30)])
                result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
                result['video'].append(self.video_feature_tool.get_video_embedding(item_id))
            yield result
            result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [], 'item_id': [],
                      "video": [], 'feature_sizes': self.FEATURE_SIZES,
                      'tile_word_size': self.title_feature_tool.MAX_WORD}

    def get_train_data_from_origin_file(self, video_path, title_path, interactive_file, audio_file_path):
        self.video_feature_tool.get_all_from_origin_file(video_path)
        print("video init finish")
        self.audio_feature_tool.get_all_from_origin_file(audio_file_path)
        print("audio init finish")
        self.title_feature_tool.get_all_from_origin_file(title_path)
        print("title init finish")
        user_action_list = self.user_interactive_tool.get_all_from_origin_file(interactive_file)
        print("user action init finish")
        train_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                        'item_id': [], "video": [], "audio": [], 'feature_sizes': self.FEATURE_SIZES,
                        'tile_word_size': self.title_feature_tool.MAX_WORD}
        val_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                      'item_id': [], "video": [], "audio": [], 'feature_sizes': self.FEATURE_SIZES,
                      'tile_word_size': self.title_feature_tool.MAX_WORD}
        val_count = 0
        for user in user_action_list:
            if random.random() > 0.95 and val_count < 100000:
                result = val_result
            else:
                result = train_result
            user_action, item_id, like, finish = user
            result['item_id'].append(item_id)
            result['like'].append(like)
            result['finish'].append(finish)
            result['index'].append(user_action)
            result['value'].append([1 for _ in user_action])
            title_list = self.title_feature_tool.get(item_id)
            result['title'].append([title_list[i] if i < len(title_list) else 0 for i in range(30)])
            result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
            result['video'].append(self.video_feature_tool.get(item_id))
            result['audio'].append(self.audio_feature_tool.get(item_id))
        return train_result, val_result


if __name__ == "__main__":
    # video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
    # title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
    # user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
    video_db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/video.db"
    title_feature_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/title.db"
    user_db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/user_test.db"
    count = 0
    for i in DataPreprocessor(video_db_path, user_db_path, title_feature_path).get_train_data():
        fp = open("/home/yuanjun/code/Bytedance_ICME_challenge/track2/submit_jsons/%s.json" % count, 'w')
        json.dump(i, fp)
        fp.close()
        count += 1
