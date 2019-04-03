import torch
from torch.utils.data import Dataset, DataLoader
import json
from data_analy.user_interactive import UserInteractiveTool


class MyDataSet(Dataset):

    def __init__(self, loader, train=True):
        self.loader = loader
        if train:
            self.user_list = self.loader.train_user_list
        else:
            self.user_list = self.loader.val_user_list

    def __getitem__(self, index):
        # result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [],
        #           'item_id': [], "video": [], "audio": [], 'feature_sizes': self.loader.FEATURE_SIZES,
        #           'tile_word_size': self.loader.title_feature_tool.MAX_WORD}
        user = self.user_list[index]
        user_action, item_id, like, finish = json.loads(user)
        user_action.append(item_id % UserInteractiveTool.ITEM_EMBEDDING_SIZE)
        # result['item_id'].append(item_id)
        # result['like'].append(like)
        # result['finish'].append(finish)
        # result['index'].append(user_action)
        # result['value'].append([1 for _ in user_action])
        title_list = json.loads(self.loader.title_feature_tool.get(item_id))
        # result['title'].append([int(title_list[i]) if i < len(title_list) else 0 for i in range(30)])
        # result['title_value'].append([1 if i < len(title_list) else 0 for i in range(30)])
        video_list = json.loads(self.loader.video_feature_tool.get(item_id))
        # result['video'].append(video_list if len(video_list) == 128 else [0 for _ in range(128)])
        # result['audio'].append(json.loads(self.loader.audio_feature_tool.get(item_id)))
        return [like, finish, user_action, [1 for _ in user_action],
                [int(title_list[i]) if i < len(title_list) else 0 for i in range(30)],
                [1 if i < len(title_list) else 0 for i in range(30)], item_id,
                video_list if len(video_list) == 128 else [0 for _ in range(128)],
                json.loads(self.loader.audio_feature_tool.get(item_id))]

    def __len__(self):
        return len(self.user_list)


if __name__ == "__main__":
    from .data_preprocessor import DataPreprocessor

    loader = DataPreprocessor()
    loader.get_origin_train_data()
    train_data = MyDataSet(loader.train_user_list)
    train_loader = DataLoader(train_data, batch_size=50000, shuffle=True, num_workers=4)

    test_data = MyDataSet(loader.val_user_list)
    test_loader = DataLoader(train_data, batch_size=50000, shuffle=True, num_workers=4)
    for i in train_loader:
        print(i)
