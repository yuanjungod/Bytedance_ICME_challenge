import json
import sqlite3
import time


class VideoFeature(object):

    def __init__(self, db_path):

        self.video_connect = sqlite3.connect(db_path)
        self.cursor = self.video_connect.cursor()

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

    def get_video_embedding(self, item_id):
        start = time.time()
        sql = "SELECT * FROM VIDEO WHERE id=%s" % item_id
        result = list()
        cursor = self.cursor.execute(sql)
        for row in cursor:
            result = json.loads(row[1])
            if len(result) != 128:
                result = [0 for _ in range(128)]
        # print("consume: %s" % (time.time() - start))
        return result


if __name__ == "__main__":
    # VideoFeature().insert("/Volumes/Seagate Expansion Drive/byte/track2/track2_video_features.txt")
    db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
    video_feature = VideoFeature(db_path)
    print(video_feature.get_video_embedding(123))
