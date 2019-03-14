from data_io.data_preprocessor import DataPreprocessor
from data_analy.user_interactive import UserInteractiveTool
import time
from common.logger import logger


video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features.txt"
title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt"
interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt"
audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features.txt"
data_prepro_tool = DataPreprocessor()

iter_data = data_prepro_tool.get_train_data_from_origin_file(video_path, title_path, interactive_file, audio_file_path)

