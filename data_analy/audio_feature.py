import json
import sqlite3
import time


class AudioFeatureTool(object):

    def __init__(self, audio_db):
        self.audio_dict = dict()
        if audio_db is not None:
            self.db_client = sqlite3.connect(audio_db)
            self.cursor = self.db_client.cursor()
        self.result_dict = None

    def create_table(self):
        sql = '''CREATE TABLE AUDIO
       (ID INT PRIMARY KEY     NOT NULL,
        ITEM_ID INT NOT NULL,
        AUDIO_FEATURE TEXT NOT NULL);'''
        self.cursor.execute(sql)
        self.db_client.commit()

    def add_index(self):
        sql = "CREATE INDEX ITEM_ID ON AUDIO (ITEM_ID);"
        self.cursor.execute(sql)
        self.db_client.commit()

    def insert(self, audio_feature_path):
        count = 0

        video_file = open(audio_feature_path, 'r')
        line = video_file.readline()
        while line:

            # print(line)
            item = json.loads(line)
            sql = "INSERT INTO AUDIO (ID, ITEM_ID, AUDIO_FEATURE) VALUES ('%s', '%s', '%s')" % (
                count, item["item_id"], json.dumps(item["audio_feature_128_dim"]))
            self.cursor.execute(sql)
            line = video_file.readline()
            count += 1
            if count % 100000 == 0:
                self.db_client.commit()
                print(count)
        self.db_client.commit()
        video_file.close()

    def get_all_from_origin_file(self, audio_file_path):
        audio_file = open(audio_file_path, 'r')
        line = audio_file.readline()
        count = 0
        while line:
            count += 1
            if count % 200000 == 0:
                print("audio", count, len(self.audio_dict))
            # print(line)
            item = json.loads(line)
            self.audio_dict[item["item_id"]] = json.dumps(item["audio_feature_128_dim"])
            line = audio_file.readline()
        audio_file.close()
        return self.audio_dict

    def get(self, item_id):
        return self.audio_dict.get(item_id, json.dumps([0 for _ in range(128)]))

    def get_from_db(self, item_id):
        start = time.time()
        sql = "SELECT * FROM AUDIO WHERE ITEM_ID=%s" % item_id
        cursor = self.cursor.execute(sql)
        # print("user consume", time.time() - start)
        # print(cursor)
        result = None
        for row in cursor:
            result = json.loads(row[2])
        if result is None:
            return [0 for _ in range(128)]
        else:
            return result

    @classmethod
    def save_origin_to_json_file(cls, audio_feature_path):
        audio_file = open(audio_feature_path, 'r')
        line = audio_file.readline()
        count = 0
        file_count = 0
        video_dict = dict()
        while line:
            count += 1
            if count % 500000 == 0:
                print("video", count, len(video_dict))
            if count % 500000 == 0:
                with open("/Volumes/Seagate Expansion Drive/byte/track2/track2_audio_features_%s.json" % file_count, "w") as f:
                    json.dump(video_dict, f)
                file_count += 1
                video_dict = dict()
                # break
            # print(line)
            item = json.loads(line)
            video_dict[item["item_id"]] = json.dumps(item["audio_feature_128_dim"])

            line = audio_file.readline()
        audio_file.close()
        with open("/Volumes/Seagate Expansion Drive/byte/track2/track2_audio_features_%s.json" % file_count, "w") as f:
            json.dump(video_dict, f)
        file_count += 1

    def get_all_from_json_file(self, video_json_file_list):
        self.result_dict = dict()
        for video_json_file in video_json_file_list:
            with open(video_json_file) as f:
                print(f)
                video_dict = json.load(f)
                for key, value in video_dict.items():
                    self.result_dict[key] = value
        # f.close()
        return self.result_dict


if __name__ == "__main__":
    audio_tool = AudioFeatureTool("/Volumes/Seagate Expansion Drive/byte/track2/audio.db")
    # audio_tool.add_index()
    # audio_tool.create_table()
    # audio_tool.insert("/Users/quantum/Downloads/track2_audio_features.txt")
    print(audio_tool.get_from_db(123))
    # audio_tool.save_origin_to_json_file("/Users/quantum/Downloads/track2_audio_features.txt")
# 3989594