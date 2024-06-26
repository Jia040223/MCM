import torch

from prob_calculate import *
import pandas as pd
import random
from train import *
from mytools import *
from parameter_get import *
from dataloader import add_p_column
from T_Test import  t_test_by_momentums
from ROC import *

M1 = 0.75
M2 = 0.10
M3 = 0.15
M4 = 0.001

name_list_path = './Data/'
file_name_df = pd.read_csv(name_list_path + 'Wimbledon_featured_matches.csv')
file_name_list = file_name_df['match_id'].unique().tolist()

def Init_Match_Prob(player1, player2):
    sets = (player1, player2)
    prob_map = Get_Match_Init_Probs()

    return  prob_map[sets]

if __name__ == "__main__":

    # 设定循环次数
    total_iterations = 5
    current_iteration = 0

    label = []
    ps = []

    for name in file_name_list:
        '''
        # 随机判断是否跳出循环
        if current_iteration == total_iterations or random.random() < 0.8 :  # 以10%的概率跳出循环
            print("跳出循环")
            continue
        # 更新迭代次数
        current_iteration += 1
        '''

        if name != "2023-wimbledon-1701":
            continue

        Momentums = []
        Win_Probability = []
        p1s = []
        p2s = []

        data_frame = pd.read_csv('./Data/' + name + '.csv')
        win_data = data_frame[['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_score', 'p2_score']].to_numpy()
        point_victor = data_frame["point_victor"]
        serves = data_frame["server"]

        p1_momentum = data_frame["p1_momentum"]
        p2_momentum = data_frame["p2_momentum"]
        p1_physStrength = data_frame["cal_m_1"]
        p2_physStrength = data_frame["cal_m_2"]

        momentum_pf = p1_momentum - p2_momentum
        p_physStrength = p1_physStrength - p2_physStrength

        columns_to_remove = list(data_frame.columns[-6:]) + list(data_frame.columns[:7]) + [data_frame.columns[15]]

        column_to_extract = 'point_victor'
        point_victor = data_frame[column_to_extract].to_numpy()

        length = len(win_data)

        player1 = data_frame["player1"][0]
        player2 = data_frame["player2"][0]

        p1 = p2 = Init_Match_Prob(player1, player2)[0] - 0.01
        orginal_p1 = p1
        data_frame = data_frame.apply(add_p_column, axis=1)
        features = data_frame.drop(columns=columns_to_remove).to_numpy()

        Momentum_Caculator = MomentumCaculater(p1, p2, 0, 0.33)

        point1 = point2 = game1 = game2 = set1 = set2 = 0
        T = serves[0] - 1
        model1 = model2 = LSTMModel(38, 32, 2, 2)
        model1.load_state_dict(torch.load('model_parameters.pth'))
        model2.load_state_dict(torch.load('model_parameters.pth'))
        model1.cuda()
        model2.cuda()

        features = torch.FloatTensor(features).cuda()

        serve_score = receive_score = 0
        server_total = receive_total = 0

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

            if serves[i] != (T + 1):
                T = serves[i] - 1
                Momentum_Caculator.UpdateT(T)

            if win_data[i][4] != set1 or win_data[i][5] != set2:
                set1 = win_data[i][0]
                set2 = win_data[i][1]
                Momentum_Caculator.UpdateSet(set1, set2)

            if T == 0:
                serve_score += (point_victor[i] == 1)
                server_total += 1
            else:
                receive_score += (point_victor[i] == 1)
                receive_total += 1

            Momentum_Caculator.GetLeverage(point1, point2, T, (point_victor[i] == 1))
            momentum, _ = Momentum_Caculator.GetMomentum()
            Momentums.append(momentum * 100 + momentum_pf[i]/3)
            win_rate = Momentum_Caculator.PredictPro(point1, point2, T)
            #print(game1, game2, set1, set2, win_rate, p1, p2)
            Win_Probability.append(win_rate)

            ''' Update P1, P2'''
            model1.eval()
            model2.eval()
            feature = features[i].unsqueeze(0).permute(1,0)

            if T == 0 :
                serve_rate = serve_score/ server_total
                finetune_p1 = model1(feature)[0][0].item()
                #print(p1, finetune_p1, serve_rate)
                if(serve_rate > 0.35 and serve_rate < 0.65):
                    p1 = p1 * M1 + finetune_p1 * M2 + serve_rate * M3 + p_physStrength[i] * M4
                else:
                    p1 = p1 * (1-M2) + finetune_p1 * M2 + + p_physStrength[i] * M4
                #ps.append(p1)
            else:
                receive_rate = receive_score/ receive_total
                finetune_p2 = model2(feature)[0][0].item()
                #print(p2, finetune_p2, receive_rate)
                if (receive_rate > 0.35 and receive_rate < 0.65):
                    p2 = p2 * M1 + finetune_p2 * M2 + receive_rate * M3 + p_physStrength[i] * M4
                else:
                    p2 = p2 * (1 - M2) + finetune_p2 * M2 + p_physStrength[i] * M4
                #ps.append(p2)

            #label.append(point_victor[i] - 1)
            Momentum_Caculator.UpdateProbability(p1, p2)

        '''
        if name == "2023-wimbledon-1701":
            Draw_Filled_Line_Graph(Momentum_Caculator.Leverages, "Points", "Leverages", "Leverages over Points" )
            Draw_Filled_Line_Graph(Momentums, "Points", "Momentums", "Momentums over Points")
            Draw_Line_Graph(Win_Probability, "Points", "Win Rate", "Win Rate over Points")
            Draw_Line_Graph(p1s, "Points", "P1", "P1 over Points")
            Draw_Line_Graph(p2s, "Points", "P2", "P2 over Points")

        t_results = t_test_by_momentums(Momentums, data_frame, max(Momentums)/2, min(Momentums)/2)

        # 创建 DataFrame 保存 t 检验结果
        result_df = pd.DataFrame(columns=['Column', 't_statistic', 'p-value', 'significant difference'])
        result_list = []
        # 遍历 t_results，将结果添加到列表
        for column, result in t_results.items():
            is_significant = result["p_value"] < 0.05
            result_list.append(
                {'Column': column, 't_statistic': result["t_statistic"], 'p-value': result["p_value"], 'significant difference': is_significant})

        # 将列表转换为 DataFrame
        result_df = pd.DataFrame(result_list)
        result_df.to_csv('./T_Test/' + name + '.csv', index=False)

        Add_csv_column('./Data/' + name + '.csv', Momentums, "Momentums", './Splite_Data/' + name + '.csv')
        '''

        # 保存结果到 CSV 文件
        Draw_Line_Graph(Win_Probability, "Points", "Win Rate", "Win Rate over Points")
        Draw_Filled_Line_Graph(Momentums, "Points", "Momentums", "Momentums over Points")
        save_to_csv(Momentums, Win_Probability, "./Sensitivity Analysis/result" + str(orginal_p1) + '.csv' )


        '''
        final_win = Judge_final_win(data_frame)
        ps.extend(Win_Probability)
        tmp = [final_win for _ in Momentums]
        label.extend(tmp)
        
    print(label)
    print(ps)
    show_ROC(label, ps)
    '''