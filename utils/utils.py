import os
import random


def get_dataset_path_list(dataset_path, sub_str=None):
    dataset_path_list = []
    for root_path, dir_names, file_names in os.walk(dataset_path):
        for file_name in file_names:
            file_path = os.path.join(root_path, file_name)
            if sub_str:
                if file_path.find(sub_str) == -1:
                    continue
            dataset_path_list.append(file_path)

    return dataset_path_list


def rand_train_data(result):
    index_list = list(range(len(result["index"])))
    random.shuffle(index_list)
    result_index = [result["index"][i] for i in index_list]
    result_value = [result["value"][i] for i in index_list]
    result_video = [result["video"][i] for i in index_list]
    result_title = [result["title"][i] for i in index_list]
    result_title_value = [result["title_value"][i] for i in index_list]
    result_like = [result["like"][i] for i in index_list]
    result_finish = [result["finish"][i] for i in index_list]
    result['index'] = result_index
    result['value'] = result_value
    result['video'] = result_video
    result['title'] = result_title
    result['title_value'] = result_title_value
    result['like'] = result_like
    result['finish'] = result_finish
    return result


