import json


class FaceFeature(object):

    def __init__(self, face_feature_path):
        face_feature_file = open(face_feature_path)
        self.face_feature_dict = dict()
        for line in face_feature_file.readlines():
            face = json.loads(line)
            self.face_feature_dict[face["item_id"]] = face["face_attrs"]
        face_feature_file.close()

    def get_face_feature(self, item_id):

        return self.face_feature_dict.get(item_id, list())
