import json
import sqlite3
import time


class TitleAnalyTool(object):

    MAX_WORD = 134416

    def __init__(self, title_db_path):
        # /Volumes/Seagate Expansion Drive/byte/track2/track2_title.txt
        if title_db_path is not None:
            self.db_client = sqlite3.connect(title_db_path)
            self.cursor = self.db_client.cursor()
        self.result_dict = None

    def create_table(self):
        sql = '''CREATE TABLE TITLE
               (ITEM_ID INT PRIMARY KEY     NOT NULL,
                TITLE_FEATURES TEXT     NOT NULL);'''
        self.cursor.execute(sql)
        self.db_client.commit()

    def insert(self, title_feature_path):
        count = 0

        title_file = open(title_feature_path, 'r')
        line = title_file.readline()
        while line:
            count += 1
            if count % 100000 == 0:
                print(count)
            # print(line)
            item = json.loads(line)
            sql = "INSERT INTO TITLE (ITEM_ID, TITLE_FEATURES) VALUES ('%s', '%s')" % (
                item["item_id"], json.dumps(list(item["title_features"].keys())))
            self.cursor.execute(sql)
            line = title_file.readline()
            count += 1
            if count % 100000 == 0:
                self.db_client.commit()
                print(count)
        title_file.close()

    def get_all_from_db(self):
        start = time.time()
        self.result_dict = dict()
        sql = "SELECT * FROM TITLE"
        cursor = self.cursor.execute(sql)
        for row in cursor:
            # print(row[0], row[1])
            self.result_dict[int(row[0])] = [int(i) for i in json.loads(row[1])]
        print("title consume", time.time() - start)

    def get_from_db(self, item_id):
        # start = time.time()
        self.result_dict = dict()
        sql = "SELECT * FROM TITLE WHERE ITEM_ID=%s" % item_id
        cursor = self.cursor.execute(sql)
        result = list()
        for row in cursor:
            # print(row[0], row[1])
            result = [int(i) for i in json.loads(row[1])]
        return result

    def get_all_from_origin_file(self, file_path):
        title_file = open(file_path, 'r')
        self.result_dict = dict()
        line = title_file.readline()
        count = 0
        while line:
            count += 1
            if count % 1000000 == 0:
                print(count)
            item = json.loads(line)
            self.result_dict[int(item["item_id"])] = json.dumps(list(item["title_features"].keys()))
            line = title_file.readline()
        title_file.close()
        return self.result_dict

    def get(self, item_id):
        if self.result_dict is None:
            print("Please load data")
            exit()
        if item_id not in self.result_dict:
            # print("title wrong!!!!!!")
            return json.dumps(list())
        # print("title OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self.result_dict.get(item_id)

    def get_title_feature_size(self):
        start = time.time()
        sql = "SELECT * FROM TITLE"
        max_id = 0
        cursor = self.cursor.execute(sql)
        count = 0
        for row in cursor:
            count += 1
            result = [int(i) for i in json.loads(row[1])]
            result.append(max_id)
            max_id = max(result)
            if count % 10000 == 0:
                print(count, max_id)
        print("consume: %s" % (time.time() - start))
        print(max_id)
        return max_id


if __name__ == "__main__":
    title_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
    title_tool = TitleAnalyTool(title_db_path)
    # title_tool.create_table()
    # title_tool.insert("/Volumes/Seagate Expansion Drive/byte/track2/track2_title.txt")
    print(title_tool.get_title_feature_size())

