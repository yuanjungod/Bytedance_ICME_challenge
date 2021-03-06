from data_analy.video_feature import *
from data_analy.user_interactive import *
from data_analy.title_analy import *
from data_analy.audio_feature import *
import json
import random

# random.seed(941)


class DataPreprocessor(object):

    FIELD_SIZE = 9
    FEATURE_SIZES = [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20]
    MAX_TITLE_SIZE = 30

    def __init__(self, video_db_path=None, user_db_path=None, title_feature_path=None, audio_feature_path=None):
        print("init video feature")
        self.video_feature_tool = VideoFeature(video_db_path)
        print("init audio feature")
        self.audio_feature_tool = AudioFeatureTool(audio_feature_path)
        print("init title feature")
        self.title_feature_tool = TitleAnalyTool(title_feature_path)
        print("init user feature")
        self.user_interactive_tool = UserInteractiveTool(user_db_path)
        self.user_action_list = None
        self.train_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                             'item_id': [], "video": [], "audio": [], 'feature_sizes': self.FEATURE_SIZES,
                             'tile_word_size': self.title_feature_tool.MAX_WORD}
        self.train_count = 0

        self.val_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                           'item_id': [], "video": [], "audio": [], 'feature_sizes': self.FEATURE_SIZES,
                           'tile_word_size': self.title_feature_tool.MAX_WORD}
        self.val_count = 0
        self.val_user_list = list()

    def get_train_data(self, step=10000):
        result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [], 'item_id': [],
                  "video": [], 'audio': [], 'feature_sizes': self.FEATURE_SIZES,
                  'tile_word_size': self.title_feature_tool.MAX_WORD}
        for i in range(0, self.user_interactive_tool.get_max_id("ID"), step):
            print("data loading")
            users = self.user_interactive_tool.get(i, i+step)
            for user in users:
                user_action, item_id, like, finish = user
                user_action.append(item_id % 500000)
                result['item_id'].append(item_id)
                result['like'].append(like)
                result['finish'].append(finish)
                result['index'].append(user_action)
                result['value'].append([1 for _ in user_action])
                title_list = list(self.title_feature_tool.get_from_db(item_id))
                result['title'].append([title_list[i] if i < len(title_list) else 0 for i in range(30)])
                result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
                result['video'].append(self.video_feature_tool.get_video_embedding(item_id))
                result['audio'].append(self.audio_feature_tool.get_from_db(item_id))
            yield result
            result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [], 'item_id': [],
                      "video": [], 'audio': [], 'feature_sizes': self.FEATURE_SIZES,
                      'tile_word_size': self.title_feature_tool.MAX_WORD}

    def get_train_data_from_origin_file(self, video_path, title_path, interactive_file, audio_file_path, step=800000):
        self.FIELD_SIZE = 10
        self.FEATURE_SIZES = [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, UserInteractiveTool.ITEM_EMBEDDING_SIZE]
        # self.FEATURE_SIZES = [80000, 400, 900000, 500, 5, 90000, 80000, 30, 20]
        # self.video_feature_tool.get_all_from_origin_file(video_path)
        if self.user_action_list is None:
            self.video_feature_tool.get_all_from_json_file([
                "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features_%s.json" % i for i in range(21)])
            print("video init finish")
            self.audio_feature_tool.get_all_from_json_file([
                "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features_%s.json" % i for i in range(8)
            ])
            print("audio init finish")
            self.title_feature_tool.get_all_from_origin_file(title_path)
            print("title init finish")
            # user_action_list = self.user_interactive_tool.get_all_from_origin_file(interactive_file)
            self.user_action_list = self.user_interactive_tool.get_all_from_json_file(
                "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.json")
            random.shuffle(self.user_action_list)
            print("user action init finish")
        else:
            random.shuffle(self.user_action_list)
            self.train_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                                 'item_id': [], "video": [], "audio": [], 'feature_sizes': self.FEATURE_SIZES,
                                 'tile_word_size': self.title_feature_tool.MAX_WORD}

        for user in self.user_action_list:
            if self.val_count < 500000 and random.random() > 0.998:
                self.val_user_list.append(user)
                result = self.val_result
                self.val_count += 1
            else:
                result = self.train_result
                self.train_count += 1
                if user in self.val_user_list:
                    continue
            user_action, item_id, like, finish = json.loads(user)
            user_action.append(item_id % UserInteractiveTool.ITEM_EMBEDDING_SIZE)
            result['item_id'].append(item_id)
            result['like'].append(like)
            result['finish'].append(finish)
            result['index'].append(user_action)
            result['value'].append([1 for _ in user_action])
            title_list = json.loads(self.title_feature_tool.get(item_id))
            result['title'].append([int(title_list[i]) if i < len(title_list) else 0 for i in range(30)])
            result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
            result['video'].append(json.loads(self.video_feature_tool.get(item_id)))
            result['audio'].append(json.loads(self.audio_feature_tool.get(item_id)))
            if self.train_count >= step:
                yield self.train_result, self.val_result
                self.train_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
                                     'item_id': [], "video": [], "audio": [], 'feature_sizes': self.FEATURE_SIZES,
                                     'tile_word_size': self.title_feature_tool.MAX_WORD}
                self.train_count = 0

    def get_train_data_from_origin_file2(self, video_path, title_path, interactive_file, audio_file_path):
        self.video_feature_tool.get_all_from_json_file([
            "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features_%s.json" % i for i in range(21)])
        print("video init finish")
        self.audio_feature_tool.get_all_from_json_file([
            "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features_%s.json" % i for i in range(8)
        ])
        print("audio init finish")
        self.title_feature_tool.get_all_from_origin_file(title_path)
        print("title init finish")
        # user_action_list = self.user_interactive_tool.get_all_from_origin_file(interactive_file)
        self.user_action_list = self.user_interactive_tool.get_all_from_json_file(
            "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_no_anwser.json")

        result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [], 'item_id': [],
                  "video": [], 'audio': [], 'feature_sizes': self.FEATURE_SIZES,
                  'tile_word_size': self.title_feature_tool.MAX_WORD}

        for user in self.user_action_list:
            user_action, item_id, like, finish = json.loads(user)
            user_action.append(item_id % UserInteractiveTool.ITEM_EMBEDDING_SIZE)
            result['item_id'].append(item_id)
            result['like'].append(like)
            result['finish'].append(finish)
            result['index'].append(user_action)
            result['value'].append([1 for _ in user_action])
            title_list = json.loads(self.title_feature_tool.get(item_id))
            result['title'].append([int(title_list[i]) if i < len(title_list) else 0 for i in range(30)])
            result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
            result['video'].append(json.loads(self.video_feature_tool.get(item_id)))
            result['audio'].append(json.loads(self.audio_feature_tool.get(item_id)))
        return result


if __name__ == "__main__":
    # video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
    # title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
    # user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user_test.db"
    # audio_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/audio.db"
    video_db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/video.db"
    title_feature_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/title.db"
    user_db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/user_test.db"
    audio_feature_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/audio.db"
    # count = 0
    # for i in DataPreprocessor(video_db_path, user_db_path, title_feature_path, audio_feature_path).get_train_data():
    #     # fp = open("/home/yuanjun/code/Bytedance_ICME_challenge/track2/submit_jsons/%s.json" % count, 'w')
    #     fp = open("/Volumes/Seagate Expansion Drive/byte/track2/submit_jsons/%s.json" % count, 'w')
    #     json.dump(i, fp)
    #     fp.close()
    #     count += 1
    result = DataPreprocessor(video_db_path, user_db_path, title_feature_path, audio_feature_path).get_train_data_from_origin_file2(
        None, "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt", None, None)
    fp = open("/home/yuanjun/code/Bytedance_ICME_challenge/track2/submit_jsons/result.json", 'w')
    json.dump(result, fp)
    fp.close()

