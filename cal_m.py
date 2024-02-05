import pandas as pd
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import numpy as np
name_list_path='./MCM/Data/'
__path__ = './MCM/Data/split_Data/'
file_name_df =pd.read_csv(name_list_path+'Wimbledon_featured_matches.csv')
file_name_list = file_name_df['match_id'].unique().tolist()

def cal_time(row):
    time_str = row['elapsed_time']
    parts = time_str.split(':')
    hours = (int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600) % 24
    return hours


def cal_e_func(x,b):
    m=2/(1+(np.exp(b*x)))
    return m

def age_to_target(age):
    # 年龄的最小值和最大值
    age_min = 17  # 为了避免对0取对数，我们将最小年龄设为1
    age_max = 40

    # 目标区间的范围
    target_min = 0.35
    target_max = 7

    # 自然对数函数的参数
    a = 10

    # 计算自然对数函数的值
    def log_func(x):
        return np.log(x + a)

    # 年龄映射到自然对数函数值
    age_mapped = log_func(age)

    # 将自然对数函数值映射到目标区间
    target_range = target_max - target_min
    age_mapped = (age_mapped - log_func(age_min)) / (log_func(age_max) - log_func(age_min)) * target_range + target_min

    return age_mapped

def std_dev_func(age):
    # 定义一个基础标准差
    base_std_dev = 1.0

    # 定义一个先缓后急增长的函数，这里以指数函数作为示例
    age_factor = np.exp(0.1 * age) - 1

    return base_std_dev * age_factor

def sample_from_normal_distribution(age):
    # 计算均值
    mu = age_to_target(age)  # 正态分布的均值
    sigma = 100  # 正态分布的标准差
    target_min = 0.35  # 目标范围的最小值
    target_max = 0.7  # 目标范围的最大值
    prob_min = norm.cdf(target_min, mu, sigma)  # 目标范围最小值对应的累积概率
    prob_max = norm.cdf(target_max, mu, sigma)  # 目标范围最大值对应的累积概率
    sample = norm.ppf(np.random.uniform(prob_min, prob_max), mu, sigma)
    return sample

for file_name in file_name_list:
    df=pd.read_csv(__path__+file_name+'.csv')

    df['elapsed_time'] = df.apply(cal_time, axis=1)

    p1_age = 25
    p2_age = 30
    b1 = sample_from_normal_distribution(p1_age)
    b2 = sample_from_normal_distribution(p2_age)
    df['cal_m_1'] = cal_e_func(df['elapsed_time'],b1)
    df['cal_m_2'] = cal_e_func(df['elapsed_time'],b2)

    df.to_csv(__path__+file_name+'.csv',index=False)

    # # 假设df是你的DataFrame
    # plt.figure(figsize=(10, 6))

    # # 绘制cal_m_1随elapsed_time变化的曲线图
    # plt.plot(df['elapsed_time'], df['cal_m_1'], label='cal_m_1')

    # # 绘制cal_m_2随elapsed_time变化的曲线图
    # plt.plot(df['elapsed_time'], df['cal_m_2'], label='cal_m_2')

    # plt.xlabel('elapsed_time')
    # plt.ylabel('cal_m')
    # plt.title('cal_m vs elapsed_time')
    # plt.legend()
    # plt.show()