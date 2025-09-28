import pickle
import os
import json

folder = 'dataset/customized_dataset/train'
files = [f for f in os.listdir(folder) if f.endswith("pkl")]
for file in files:
    with open(os.path.join(folder, file), 'rb') as f:
        data = pickle.load(f)
    kps = data['keypoints']
    json_dict = {}
    json_dict['kps'] = kps.tolist()
    base_filename = os.path.splitext(file)[0]
    json_filename = base_filename + ".json"
    json_file_path = os.path.join(folder, json_filename)
    with open(json_file_path, 'w') as json_f:
        json.dump(json_dict, json_f, indent=4)