from data_analy.video_feature import *
from data_analy.user_interactive import *
from data_analy.title_analy import *
import os
import pandas as pd


class DataPreprocessor(object):

    FIELD_SIZE = 9
    FEATURE_SIZES = [80000, 400, 900000, 500, 5, 90000, 80000, 30, 20]
    MAX_TITLE_SIZE = 30

    def __init__(self, video_db_path, user_db_path, title_feature_path):
        print("init video feature")
        self.video_feature_tool = VideoFeature(video_db_path)
        print("init title feature")
        self.title_feature_tool = TitleAnalyTool(title_feature_path)
        print("init user feature")
        self.user_interactive_tool = UserInteractiveTool(user_db_path)

    def get_train_data(self):
        result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                  "video": [], 'feature_sizes': self.FEATURE_SIZES, 'tile_word_size': self.title_feature_tool.MAX_WORD}
        step = 100000
        for i in range(0, self.user_interactive_tool.get_max_id("ID"), step):
            print("data loading")
            for j in range(i, i + step):
                user_action, item_id, like, finish = self.user_interactive_tool.get(j)
                result['like'].append(like)
                result['finish'].append(finish)
                result['index'].append(user_action)
                result['value'].append([1 for _ in user_action])
                title_list = list(self.title_feature_tool.get(item_id))
                result['title'].append([title_list[i] if i < len(title_list) else 0 for i in range(30)])
                result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
                result['video'].append(self.video_feature_tool.get_video_embedding(item_id))
            yield result
            result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                      "video": [], 'feature_sizes': self.FEATURE_SIZES,
                      'tile_word_size': self.title_feature_tool.MAX_WORD}


if __name__ == "__main__":
    video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
    title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
    user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
    for i in DataPreprocessor(video_db_path, user_db_path, title_feature_path).get_train_data():
        print(i)
        exit()