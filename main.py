import torch

from prob_calculate import *
import pandas as pd
from train import *

M = 0.9

if __name__ == "__main__":
    data_frame = pd.read_csv('Data/Wimbledon_featured_matches_processed.csv')
    win_data = data_frame[['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_score', 'p2_score']].to_numpy()

    columns_to_remove = list(data_frame.columns[-2:]) + list(data_frame.columns[:7])
    features = data_frame.drop(columns=columns_to_remove).to_numpy()

    column_to_extract = 'point_victor'
    point_victor = data_frame[column_to_extract].to_numpy()

    length = len(win_data)

    p1 = p2 = 0.5
    Momentum_Caculator = MomentumCaculater(p1, p2, 0, 0.33)

    point1 = point2 = game1 = game2 = set1 = set2 = 0
    T = 0
    model1 = model2 = train(p1)
    #model2 = train(p2)

    features = torch.FloatTensor(features).cuda()

    for i in range(length):
        point1 = win_data[i][4]
        point2 = win_data[i][5]
        if win_data[i][2] != game1 or win_data[i][3] != game2:
            game1 = win_data[i][2]
            game2 = win_data[i][3]
            Momentum_Caculator.UpdateGame(game1, game2)
            if T == 0:
                T = 1
                Momentum_Caculator.UpdateT(T)
            else:
                T = 0
                Momentum_Caculator.UpdateT(T)

        if win_data[i][4] != set1 or win_data[i][5] != set2:
            set1 = win_data[i][0]
            set2 = win_data[i][1]
            Momentum_Caculator.UpdateSet(set1, set2)


        Momentum_Caculator.GetLeverage(point1, point2, T, (point_victor[i] == 1))
        momentum, _ = Momentum_Caculator.GetMomentum()

        ''' Update P1, P2'''
        model1.eval()
        model2.eval()
        feature = features[i].unsqueeze(0).permute(1,0)

        if T == 0 :
            finetune_p1 = model1(feature)[0][0].item()
            p1 = p1 * M + finetune_p1 * (1 - M)
        else:
            finetune_p2 = model2(feature)[0][0].item()
            p2 = p2 * M + finetune_p2 * (1 - M)

        print("p : ", p1, "T :", T)
        print("momentum :", momentum)

        Momentum_Caculator.UpdateProbability(p1, p2)
