import pandas as pd


class UserInteractiveTool(object):

    def __init__(self, user_interactivate_path, debug=False):
        # /Volumes/Seagate Expansion Drive/byte/track2/final_track2_train.txt
        track_file = open(user_interactivate_path)
        user_interactive_dict = dict()
        user_interactive_dict['uid'] = list()
        user_interactive_dict['user_city'] = list()
        user_interactive_dict['item_id'] = list()
        user_interactive_dict['author_id'] = list()
        user_interactive_dict['item_city'] = list()
        user_interactive_dict['channel'] = list()
        user_interactive_dict['finish'] = list()
        user_interactive_dict['like'] = list()
        user_interactive_dict['music_id'] = list()
        user_interactive_dict['device_id'] = list()
        user_interactive_dict['create_time'] = list()
        user_interactive_dict['video_duration'] = list()

        read_count = 0
        line = track_file.readline()
        while line:
            read_count += 1
            if read_count > 10000 and debug:
                break
            if read_count % 10000 == 0:
                print(read_count)
            column_list = line.split("\t")
            user_interactive_dict['uid'].append(column_list[0])
            user_interactive_dict['user_city'].append(column_list[1])
            user_interactive_dict['item_id'].append(column_list[2])
            user_interactive_dict['author_id'].append(column_list[3])
            user_interactive_dict['item_city'].append(column_list[4])
            user_interactive_dict['channel'].append(column_list[5])
            user_interactive_dict['finish'].append(column_list[6])
            user_interactive_dict['like'].append(column_list[7])
            user_interactive_dict['music_id'].append(column_list[8])
            user_interactive_dict['device_id'].append(column_list[9])
            user_interactive_dict['create_time'].append(column_list[10])
            user_interactive_dict['video_duration'].append(column_list[11])
            line = track_file.readline()

        self.user_interactive_pd = pd.DataFrame(user_interactive_dict)
        del user_interactive_dict
        track_file.close()

    @classmethod
    def one_hot_video_duration(cls, duration):
        duration = int(duration)
        if 0 < duration <= 1:
            return 1
        elif 1 < duration <= 2:
            return 2
        elif 2 < duration <= 3:
            return 3
        elif 3 < duration <= 4:
            return 4
        elif 4 < duration <= 5:
            return 5
        elif 5 < duration <= 10:
            return 6
        elif 10 < duration <= 20:
            return 7
        elif 20 < duration <= 30:
            return 8
        elif 30 < duration <= 40:
            return 9
        elif 40 < duration <= 50:
            return 10
        elif 50 < duration <= 60:
            return 11
        elif duration > 60:
            return 12
        else:
            return 0

    @classmethod
    def one_hot_create_time(cls, create_time):
        create_time = int(create_time) - 53087062992
        symbol = 0 if create_time >= 0 else 12
        create_time = abs(create_time)
        one_day = 24*60*60
        if 0 < create_time <= one_day:
            return 1 + symbol
        elif 1*one_day < create_time <= 2*one_day:
            return 2 + symbol
        elif 2*one_day < create_time <= 3*one_day:
            return 3 + symbol
        elif 3*one_day < create_time <= 4*one_day:
            return 4 + symbol
        elif 4*one_day < create_time <= 5*one_day:
            return 5 + symbol
        elif 5*one_day < create_time <= 10*one_day:
            return 6 + symbol
        elif 10*one_day < create_time <= 20*one_day:
            return 7 + symbol
        elif 20*one_day < create_time <= 30*one_day:
            return 8 + symbol
        elif 30*one_day < create_time <= 40*one_day:
            return 9 + symbol
        elif 40*one_day < create_time <= 50*one_day:
            return 10 + symbol
        elif 50*one_day < create_time <= 60*one_day:
            return 11 + symbol
        elif create_time > 60*one_day:
            return 12 + symbol
        else:
            return 0

    def one_hot_features(self):
        self.user_interactive_pd["video_duration"] = self.user_interactive_pd["video_duration"].map(self.one_hot_video_duration)
        self.user_interactive_pd["create_time"] = self.user_interactive_pd["create_time"].map(self.one_hot_create_time)
        return self.user_interactive_pd


if __name__ == "__main__":
    interactive_tool = UserInteractiveTool("/Volumes/Seagate Expansion Drive/byte/track2/final_track2_train.txt")
    interactive_tool.one_hot_features()
    print(interactive_tool.user_interactive_pd.to_dict())
    interactive_tool.user_interactive_pd.to_csv("test.csv")
