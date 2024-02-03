import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np


class WimbledonDataset(Dataset):
    def __init__(self):
        csv_file_path = 'Data/Wimbledon_featured_matches_processed.csv'
        data_frame = pd.read_csv(csv_file_path)

        # 去除倒数第二列和前7列
        '''self.elapsed_time = data_frame.groupby('game_id').apply(lambda group: group.iloc[:, [3]].reset_index(drop=True).values).to_numpy()
        self.elapsed_time = [[int(time_str[0].split(":")[0]) * 3600 + int(time_str[0].split(":")[1]) * 60 + int(time_str[0].split(":")[2]) for time_str in time_list] for time_list in self.elapsed_time]
        self.point_victor = data_frame.groupby('game_id').apply(lambda group: group.iloc[:, [15]].reset_index(drop=True).values).to_numpy()
        self.features = data_frame.groupby('game_id').apply(lambda group: group.iloc[:, 7:-2].reset_index(drop=True).values).to_numpy()
        '''

        columns_to_remove = list(data_frame.columns[-2:]) + list(data_frame.columns[:7])
        column_to_extract = 'elapsed_time'
        self.elapsed_time = data_frame[column_to_extract].to_numpy()
        column_to_extract = 'point_victor'
        self.point_victor = data_frame[column_to_extract].to_numpy()
        self.features = data_frame.drop(columns=columns_to_remove).to_numpy()

        time_array = np.array(self.elapsed_time)
        self.elapsed_time = time_array.reshape(-1, 1)

        mapping = {1: np.array([1, 0]), 2: np.array([0, 1])}
        self.point_victor = np.array([mapping[val] for val in self.point_victor])

        contains_nan = np.where(np.isnan(self.features))
        print(contains_nan)

        self.len = len(self.features)

    def __getitem__(self, index):
        return torch.Tensor(self.point_victor[index]),\
                torch.FloatTensor(self.features[index]), \


    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

if __name__ == "__main__":
    test = WimbledonDataset()