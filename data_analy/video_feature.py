import json
import sqlite3
import time


class VideoFeature(object):

    def __init__(self, db_path):
        if db_path is not None:
            self.video_connect = sqlite3.connect(db_path)
            self.cursor = self.video_connect.cursor()
        self.video_dict = dict()

    def insert(self, video_feature_path):
        count = 0

        video_file = open(video_feature_path, 'r')
        line = video_file.readline()
        while line:

            # print(line)
            item = json.loads(line)
            sql = "INSERT INTO VIDEO (ID, FEATURE) VALUES ('%s', '%s')" % (
                item["item_id"], json.dumps(item["video_feature_dim_128"]))
            self.cursor.execute(sql)
            line = video_file.readline()
            count += 1
            if count % 100000 == 0:
                self.video_connect.commit()
                print(count)
        self.video_connect.commit()
        video_file.close()

    def get_all_from_origin_file(self, video_feature_path):
        video_file = open(video_feature_path, 'r')
        line = video_file.readline()
        count = 0
        while line:
            count += 1
            if count % 500000 == 0:
                print("video", count, len(self.video_dict))
                # break
            # print(line)
            item = json.loads(line)
            self.video_dict[item["item_id"]] = json.dumps(item["video_feature_dim_128"])
            line = video_file.readline()
        video_file.close()
        return self.video_dict

    @classmethod
    def save_origin_to_json_file(cls, video_feature_path):
        video_file = open(video_feature_path, 'r')
        line = video_file.readline()
        count = 0
        file_count = 0
        video_dict = dict()
        while line:
            count += 1
            if count % 200000 == 0:
                print("video", count, len(video_dict))
            if count % 200000 == 0:
                with open("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features_%s.json" % file_count, "w") as f:
                    json.dump(video_dict, f)
                file_count += 1
                video_dict = dict()
                # break
            # print(line)
            item = json.loads(line)
            video_dict[item["item_id"]] = json.dumps(item["video_feature_dim_128"])

            line = video_file.readline()
        video_file.close()
        with open("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features_%s.json" % file_count, "w") as f:
            json.dump(video_dict, f)
        file_count += 1

    def get_all_from_json_file(self, video_json_file_list):
        for video_json_file in video_json_file_list:
            with open(video_json_file) as f:
                print(f)
                video_dict = json.load(f)
                for key, value in video_dict.items():
                    # print(type(key))
                    # exit()
                    self.video_dict[int(key)] = value
        # f.close()
        return self.video_dict

    def get(self, item_id):
        if len(self.video_dict) == 0:
            print("load video first!!!!!!")
            exit()
        if item_id not in self.video_dict:
            # print("video embedding is 0!!!!!!")
            return json.dumps([0 for _ in range(128)])
        # print("video ok!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self.video_dict.get(item_id)

    def get_video_embedding(self, item_id):
        start = time.time()
        sql = "SELECT * FROM VIDEO WHERE id=%s" % item_id
        result = list()
        cursor = self.cursor.execute(sql)
        for row in cursor:
            result = json.loads(row[1])

        # print("consume: %s" % (time.time() - start))
        if len(result) == 0:
            print("video embedding is 0!!!!!!!", item_id)
            result = [0 for _ in range(128)]
        return result


if __name__ == "__main__":
    # VideoFeature().insert("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features.txt")
    db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
    video_feature = VideoFeature(db_path)
    # print(video_feature.get_video_embedding(123))
    # video_dict = video_feature.get_all_from_origin_file("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features.txt")
    # fp = open("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features.json", "w")
    # json.dump(video_dict, fp)
    # fp.close()
    # video_feature.save_origin_to_json_file("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features.txt")
    # print(len(video_feature.video_dict))
    video_feature.get_all_from_json_file([
        "/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features_%s.json" % i for i in range(21)])
    # time.sleep(200)
