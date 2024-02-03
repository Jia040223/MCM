import pandas as pd
import numpy as np
__path__ = 'MCM/Data/'

df = pd.read_csv(__path__ + 'Wimbledon_featured_matches.csv')

def fill_na_with_proportional_values(group):
    value_counts = group['return_depth'].value_counts(normalize=True)
    values = value_counts.index
    probs = value_counts.values
    group['return_depth'] = group['return_depth'].apply(lambda x: np.random.choice(values, p=probs) if pd.isna(x) else x)
    group['return_depth'] = group['return_depth'].apply(lambda x: 0 if x == 'ND' else 1 if x == 'D' else 2)
    group['serve_depth'] = group['serve_depth'].apply(lambda x: 0 if x == 'NCTL' else 1 if x == 'CTL' else 2)
    group['serve_width'] = group['serve_width'].apply(lambda x: 0 if x == 'W' else 1 if x == 'C' else 2 if x== 'B' else 3 if x=='BW' else 4)
    group['winner_shot_type'] = group['winner_shot_type'].apply(lambda x: 1 if x == 'F' else 2 if x == 'B' else 0)
    return group

def replce_score_comparison(group):
    # print(group['p1_score'])
    # print(group['p2_score'])
    group['p1_score'] = group['p1_score'].replace({'0': 0, '15': 1, '30': 2})
    group['p2_score'] = group['p2_score'].replace({'0': 0, '15': 1, '30': 2})
    not_ad_40_or_40_40_rows = group[((group['p1_score'] == '40') & ~group['p2_score'].isin(['AD', '40'])) |
                                 ((group['p2_score'] == '40') & ~group['p1_score'].isin(['AD', '40']))]
    group.loc[not_ad_40_or_40_40_rows.index, ['p1_score', 'p2_score']] = group.loc[not_ad_40_or_40_40_rows.index, ['p1_score', 'p2_score']].replace('40', 3)

    if not ((group['p1_score'] == 3) & (group['p2_score'] == 2)|(group['p2_score'] == 3) & (group['p1_score'] == 2)).any():
        return group
    first_ad_position = group[((group['p1_score'] ==3) & (group['p2_score'] == 2)|(group['p2_score'] == 3) & (group['p1_score'] == 2))].first_valid_index()
    if not ((group['p1_score'] == '40') & (group['p2_score'] == '40')).any():
        return group
    first_forty_position = group[(group['p1_score'] == '40') & (group['p2_score'] == '40')].first_valid_index()
    group.loc[first_forty_position, ['p1_score', 'p2_score']] = group.loc[first_forty_position, ['p1_score', 'p2_score']].replace({'40': 3, '40': 3})
    # print(group['p1_score'])
    # print(group['p2_score'])
    

    after_first_forty = group.loc[first_forty_position+1:]
    ad_40_or_40_40_rows = after_first_forty[(after_first_forty['p1_score'].isin(['AD']) & after_first_forty['p2_score'].isin(['40'])) |
                                        (after_first_forty['p1_score'].isin(['40']) & after_first_forty['p2_score'].isin(['40'])) |
                                        (after_first_forty['p1_score'].isin(['40']) & after_first_forty['p2_score'].isin(['AD']))]
    # print(ad_40_or_40_40_rows)
    for idx in ad_40_or_40_40_rows.index:
        if group.loc[idx, 'p1_score'] == 'AD' and group.loc[idx, 'p2_score'] == '40':
            group.loc[idx, 'p1_score'] = group.loc[idx-1, 'p1_score'] + 1
            group.loc[idx, 'p2_score'] = group.loc[idx-1, 'p2_score']
            # print(group.loc[idx, 'p1_score'])

        # 如果'p2_score'列的值为'AD'，将其替换为上一行的值+1
        if group.loc[idx, 'p2_score'] == 'AD' and group.loc[idx, 'p1_score'] == '40':
            group.loc[idx, 'p2_score'] = group.loc[idx-1, 'p2_score'] + 1
            group.loc[idx, 'p1_score'] = group.loc[idx-1, 'p1_score']
            # print(group.loc[idx, 'p2_score'])

        # 如果'p1_score'和'p2_score'列的值都为'40'，将小的那个值更新为大的那个值
        if group.loc[idx, 'p1_score'] == '40' and group.loc[idx, 'p2_score'] == '40':
            # print(group.loc[idx-1, 'p1_score'])
            # print(group.loc[idx-1, 'p2_score'])
            group.loc[idx, 'p1_score'] = max(group.loc[idx-1, 'p1_score'], group.loc[idx-1, 'p2_score'])
            group.loc[idx, 'p2_score'] = max(group.loc[idx-1, 'p1_score'], group.loc[idx-1, 'p2_score'])
    return group



df = df.groupby('match_id').apply(fill_na_with_proportional_values).reset_index(drop=True)

df['new_game'] = (df['game_victor'].shift() > df['game_victor']).astype(int)
df['game_id'] = df['new_game'].cumsum()

df = df.groupby('game_id').apply(replce_score_comparison).reset_index(drop=True)
df = df.drop(['new_game', 'game_id'], axis=1)
df.to_csv(__path__ + 'Wimbledon_featured_matches_processed.csv', index=False)