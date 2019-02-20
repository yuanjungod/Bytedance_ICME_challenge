from data_analy.video_feature import *
from data_analy.user_interactive import *
from data_analy.title_analy import *


class DataPreprocessor(object):

    def __init__(self, video_feature_path, title_feature_path, user_interactivate_path, debug=False):
        print("init video feature")
        self.video_feature_tool = VideoFeature(video_feature_path, debug)
        print("init title feature")
        self.title_feature_tool = TitleAnalyTool(title_feature_path, debug)
        print("init user feature")
        self.user_interactive_tool = UserInteractiveTool(user_interactivate_path, debug)

    def preprocess(self):
        print("one hot begin")
        self.user_interactive_tool.one_hot_features()
        print("merge title begin")
        self.user_interactive_tool.user_interactive_pd["title"] = \
            self.user_interactive_tool.user_interactive_pd["item_id"].map(
                lambda a: self.title_feature_tool.origin_feature_dict.get(a, None))
        print("merge video begin")
        self.user_interactive_tool.user_interactive_pd["video"] = \
            self.user_interactive_tool.user_interactive_pd["item_id"].map(
                lambda a: self.video_feature_tool.video_dict.get(a, None))
        print("save begin")
        step = 100000
        for i in range(0, len(self.user_interactive_tool.user_interactive_pd), step):
            self.user_interactive_tool.user_interactive_pd[i: i+step].to_csv(
                "/Volumes/Seagate Expansion Drive/byte/track2/%s.csv" % i)
        print("save end")


if __name__ == "__main__":
    video_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features.txt"
    title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/track2_title.txt"
    user_interactive_path = "/Volumes/Seagate Expansion Drive/byte/track2/final_track2_train.txt"
    DataPreprocessor(video_feature_path, title_feature_path, user_interactive_path, False).preprocess()