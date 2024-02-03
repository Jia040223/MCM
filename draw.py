import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
name_list_path='./Data/'
__path__ = './Data/'
file_name_df =pd.read_csv(name_list_path+'Wimbledon_featured_matches.csv')
file_name_list = file_name_df['match_id'].unique().tolist()


point_value = 1
game_value = 5
set_value = 30
consecutive_wins_value = 90
a,b,c=0.1,0.3,0.8
steepness = 0.05
ace_value = 0.75
double_fault_value = -0.65
winner_value = 0.15
unforced_error_value = -0.35
break_point_value = 2.5
break_pt_won = 7.5
net_point_value = 0.3

steepness = 0.17

def count_consecutive_wins(df, column_name, new_column_name_1, new_column_name_2):
    df_temp = df[df[column_name] != 0]
    df_temp['new_win'] = (df_temp[column_name] != df_temp[column_name].shift()).cumsum()
    df_temp['consecutive_wins'] = df_temp.groupby(['new_win', column_name]).cumcount() + 1
    df[new_column_name_1] = df_temp[df_temp[column_name] == 1]['consecutive_wins']
    df[new_column_name_2] = df_temp[df_temp[column_name] == 2]['consecutive_wins']
    df[new_column_name_1].fillna(0, inplace=True)
    df[new_column_name_2].fillna(0, inplace=True)
    first_non_zero_idx_1 = df[new_column_name_1].ne(0).idxmax()
    if first_non_zero_idx_1 != 0:
        df.loc[first_non_zero_idx_1:, new_column_name_1] = df.loc[first_non_zero_idx_1:, new_column_name_1].replace(0, np.nan)
    df[new_column_name_1].fillna(0, inplace=True)

    first_non_zero_idx_2 = df[new_column_name_2].ne(0).idxmax()
    if first_non_zero_idx_2 != 0:
        df.loc[first_non_zero_idx_2:, new_column_name_2] = df.loc[first_non_zero_idx_2:, new_column_name_2].replace(0, np.nan)
    df[new_column_name_2].fillna(0, inplace=True)
    return df

def cal_sigmoid(a,b,c,x_1,x_2,x_3):
    return consecutive_wins_value/(1+np.exp(-steepness*(a*x_1+b*x_2+c*x_3)))

def cal_momentum(row):
    p1_momentum = cal_sigmoid(a, b, c, row['consecutive_point_wins_1'], row['consecutive_game_wins_1'], row['consecutive_set_wins_1'])
    p2_momentum = cal_sigmoid(a, b, c, row['consecutive_point_wins_2'], row['consecutive_game_wins_2'], row['consecutive_set_wins_2'])
    # 这里假设你想要将p1_momentum和p2_momentum相加
    p1_momentum += row['p1_points_won'] * point_value + row['p1_cumulative_wins'] * game_value + row['p1_sets'] * set_value
    p2_momentum += row['p2_points_won'] * point_value + row['p2_cumulative_wins'] * game_value + row['p2_sets'] * set_value
    if row['p1_ace'] == 1:
        p1_momentum += ace_value
    if row['p2_ace'] == 1:
        p2_momentum += ace_value
    if row['p1_double_fault'] == 1:
        p1_momentum -= double_fault_value
    if row['p2_double_fault'] == 1:
        p2_momentum -= double_fault_value
    if row['p1_winner'] == 1:
        p1_momentum += winner_value
    if row['p2_winner'] == 1:
        p2_momentum += winner_value
    if row['p1_unf_err'] == 1:
        p1_momentum -= unforced_error_value
    if row['p2_unf_err'] == 1:
        p2_momentum -= unforced_error_value
    if row['p1_break_pt'] == 1:
        p1_momentum += break_point_value
    if row['p2_break_pt'] == 1:
        p2_momentum += break_point_value
    if row['p1_break_pt_won'] == 1:
        p1_momentum += break_pt_won
    if row['p2_break_pt_won'] == 1:
        p2_momentum += break_pt_won
    if row['p1_net_pt'] == 1:
        p1_momentum += net_point_value
    if row['p2_net_pt'] == 1:
        p2_momentum += net_point_value
    return p1_momentum, p2_momentum


for file_name in file_name_list:
    df=pd.read_csv(__path__+file_name+'.csv')
    df['set_no'] = df['set_no'].astype(int)
    set_numbers = df['set_no'].unique()

    for set_no in set_numbers:
        set_df = df[df['set_no'] == set_no]
        if set_no == set_numbers[0]:  # 如果是第一盘
            df.loc[df['set_no'] == set_no, 'p1_cumulative_wins'] = set_df['p1_games']
            df.loc[df['set_no'] == set_no, 'p2_cumulative_wins'] = set_df['p2_games']
        else:  # 如果不是第一盘
            p1_initial_wins = df.loc[df['set_no'] == set_no - 1, 'p1_cumulative_wins'].iloc[-1]
            p2_initial_wins = df.loc[df['set_no'] == set_no - 1, 'p2_cumulative_wins'].iloc[-1]
            df.loc[df['set_no'] == set_no, 'p1_cumulative_wins'] = p1_initial_wins + set_df['p1_games']
            df.loc[df['set_no'] == set_no, 'p2_cumulative_wins'] = p2_initial_wins + set_df['p2_games']




    # 统计连胜点
    df = count_consecutive_wins(df, 'point_victor', 'consecutive_point_wins_1', 'consecutive_point_wins_2')

    # 统计连胜局
    df = count_consecutive_wins(df, 'game_victor', 'consecutive_game_wins_1', 'consecutive_game_wins_2')

    # 统计连胜盘
    df = count_consecutive_wins(df, 'set_victor', 'consecutive_set_wins_1', 'consecutive_set_wins_2')

    p1_momentum = 0.0
    p2_momentum = 0.0

    df['p1_momentum'], df['p2_momentum'] = zip(*df.apply(cal_momentum, axis=1))
    df = df.drop(['speed_mph','p1_cumulative_wins','p2_cumulative_wins','consecutive_point_wins_1', 'consecutive_game_wins_1', 'consecutive_set_wins_1', 'consecutive_point_wins_2', 'consecutive_game_wins_2', 'consecutive_set_wins_2'], axis=1)
    df.to_csv(__path__+file_name+'.csv', index=False)

# plt.figure(figsize=(10,6))
# plt.plot(df['point_no'], df['p1_momentum'], label='Player 1 Momentum')
# plt.plot(df['point_no'], df['p2_momentum'], label='Player 2 Momentum')
# plt.xlabel('Point Number')
# plt.ylabel('Momentum')
# plt.title('Momentum of Player 1 and Player 2 over Point Number')
# plt.legend()
# plt.show()

