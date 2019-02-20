import json


class FaceFeature(object):

    def __init__(self, face_feature_path):
        face_feature_file = open(face_feature_path)
        face_feature_dict = dict()
        for line in face_feature_file.readlines():
            face = json.loads(line)
            face_feature_dict[face["item_id"]] = face["face_attrs"]
