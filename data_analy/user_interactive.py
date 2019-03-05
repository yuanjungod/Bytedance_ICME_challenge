import sqlite3
import time
import json


class UserInteractiveTool(object):

    ITEM_EMBEDDING_SIZE = 1000000

    def __init__(self, db_path):
        # /Volumes/Seagate Expansion Drive/byte/track2/final_track2_train.txt
        if db_path is not None:
            self.db_client = sqlite3.connect(db_path)
            self.cursor = self.db_client.cursor()
        self.user_interactivate_list = list()

        # user_interactive_dict = dict()
        # user_interactive_dict['uid'] = list()
        # user_interactive_dict['user_city'] = list()
        # user_interactive_dict['item_id'] = list()
        # user_interactive_dict['author_id'] = list()
        # user_interactive_dict['item_city'] = list()
        # user_interactive_dict['channel'] = list()
        # user_interactive_dict['finish'] = list()
        # user_interactive_dict['like'] = list()
        # user_interactive_dict['music_id'] = list()
        # user_interactive_dict['device_id'] = list()
        # user_interactive_dict['create_time'] = list()
        # user_interactive_dict['video_duration'] = list()

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

    def create_table(self):
        sql = '''CREATE TABLE USER_TEST
       (ID INT PRIMARY KEY     NOT NULL,
        UID INT     NOT NULL, 
        USER_CITY INT     NOT NULL,
        ITEM_ID INT     NOT NULL,
        AUTHOR_ID INT     NOT NULL,
        ITEM_CITY INT     NOT NULL,
        CHANNEL INT     NOT NULL,
        FINISH INT     NOT NULL,
        LIKE INT     NOT NULL,
        MUSIC_ID INT     NOT NULL,
        DEVICE_ID INT     NOT NULL,
        CREATE_TIME INT     NOT NULL,
        VIDEO_DURATION INT     NOT NULL);'''
        self.cursor.execute(sql)
        self.db_client.commit()

    def insert(self, user_interactivate_path):
        read_count = 0
        track_file = open(user_interactivate_path)
        line = track_file.readline()
        while line:
            column_list = line.split("\t")
            sql = "INSERT INTO USER_TEST (ID, UID, USER_CITY, ITEM_ID, AUTHOR_ID, ITEM_CITY, CHANNEL, FINISH, LIKE, " \
                  "MUSIC_ID, DEVICE_ID, CREATE_TIME, VIDEO_DURATION) VALUES (" \
                  "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (
                read_count, column_list[0], column_list[1], column_list[2], column_list[3], column_list[4],
                column_list[5], column_list[6], column_list[7], column_list[8], column_list[9],
                self.one_hot_create_time(column_list[10]), self.one_hot_video_duration(column_list[11]))
            self.cursor.execute(sql)
            line = track_file.readline()
            read_count += 1
            if read_count % 10000 == 0:
                self.db_client.commit()
                print(read_count)
        track_file.close()

    def get_all_from_origin_file(self, user_interactivate_path):
        track_file = open(user_interactivate_path)
        line = track_file.readline()
        count = 0
        while line:
            count += 1
            if count % 1000000 == 0:
                print(count, len(self.user_interactivate_list))
                # break
            column_list = line.split("\t")
            # print(column_list)
            # exit()
            current_result = list()
            finish = int(column_list[6])
            like = int(column_list[7])
            item_id = int(column_list[2])
            column_list[10] = self.one_hot_create_time(column_list[10])
            column_list[11] = self.one_hot_video_duration(column_list[11])
            for i in range(len(column_list)):
                if i not in [2, 6, 7]:
                    current_result.append(int(column_list[i])+1)
            # print([current_result, item_id, like, finish])
            # exit()
            self.user_interactivate_list.append(json.dumps([current_result, item_id, like, finish]))
            line = track_file.readline()
        track_file.close()
        return self.user_interactivate_list

    def get_all_from_json_file(self, user_interactive_file):
        f = open(user_interactive_file, "r")
        print(f)
        self.user_interactivate_list = json.load(f)
        return self.user_interactivate_list

    def get(self, record_id_1, record_id_2):
        start = time.time()
        sql = "SELECT * FROM USER WHERE id>=%s and id < %s" % (record_id_1, record_id_2)
        result = list()
        cursor = self.cursor.execute(sql)
        # print("user consume", time.time() - start)
        # print(cursor)
        for row in cursor:
            # print("shit")
            # print(row)
            record = row
            # result["item_id"] = row[1]
            # print("consume: %s" % (time.time() - start))
            item_id = record[3]
            finish = record[7]
            like = record[8]
            for i in range(len(record)):
                if i not in [0, 3, 7, 8]:
                    result.append(record[i]+1)
            yield result, item_id, like, finish
            result = list()

    def get_max_id(self, name):
        start = time.time()
        sql = "SELECT max(%s) FROM USER" % name
        result = list()
        cursor = self.cursor.execute(sql)
        for row in cursor:
            print(name, row)
            result.append(row)
            # result["item_id"] = row[1]
        # print("consume: %s" % (time.time() - start))
        return result[0][0]

    def get_features_size(self):
        feature_size_list = list()
        for i in ["UID", "USER_CITY", "AUTHOR_ID", "ITEM_CITY", "CHANNEL",  "MUSIC_ID",
                  "DEVICE_ID", "CREATE_TIME", "VIDEO_DURATION"]:
            max_id = self.get_max_id(i)
            feature_size_list.append(max_id)
        print(feature_size_list)


if __name__ == "__main__":
    import time
    db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/user.db"
    interactive_tool = UserInteractiveTool(db_path=db_path)
    # interactive_tool.create_table()
    # interactive_tool.create_table()
    # interactive_tool.insert("/Volumes/Seagate Expansion Drive/byte/track2/final_track2_test_no_anwser.txt")
    # print(interactive_tool.get_max_id())
    # print(interactive_tool.get(100))
    # print(interactive_tool.get_features_size())
    # for i in ["ID", "UID", "USER_CITY", "ITEM_ID", "AUTHOR_ID", "ITEM_CITY", "CHANNEL", "FINISH", "LIKE", "MUSIC_ID",
    #           "DEVICE_ID", "CREATE_TIME", "VIDEO_DURATION"]:
    #     interactive_tool.get_max_id(i)
    user_interactivate_list = interactive_tool.get_all_from_origin_file(
        "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_test_no_anwser.txt")
    f = open("/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_no_anwser.json", "w")
    json.dump(user_interactivate_list, f)
    # f.close()
    # interactive_tool.get_all_from_json_file("/Volumes/Seagate Expansion Drive/byte/track2/final_track2_train.json")
    # print(len(interactive_tool.user_interactivate_list))
    # time.sleep(100)
    # print(interactive_tool.get_max_id("ITEM_ID"))
    # interactive_tool.get_all_from_origin_file("/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt")