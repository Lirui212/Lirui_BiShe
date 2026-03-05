import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.seterr(divide='ignore',invalid='ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ----------------------------读数据----------------------------------

data = np.load('whole_data.npy', allow_pickle=True).item()

x_train = np.array(data['train_sample'].reshape(-1, 2081))
y_train = np.array(data['train_label'])
x_test = np.array(data['test_sample'].reshape(-1, 2081))
y_test = np.array(data['test_label'])

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

all_data = np.concatenate((x, y), axis=1)

# print(all_data.shape)
sensor = all_data[:, 2048:2063]  # 抽取传感器数据,15维
condition = all_data[:, 2063:2081]  # 抽取工况数据, 前13维为压力, 后5维为频率
label = all_data[:, 2081:]  # 故障标签, 16维

condition_dictionary = np.unique(condition, axis=0)  # 工况字典, 包含不同类别的工况编码, 目前有18种工况组合
sensor_dictionary = np.unique(sensor, axis=0)  # 传感器字典, 包含不同传感器编码
label_dictionary = np.unique(label, axis=0)  # 标签字典
# print(sensor_dictionary)

# 1.先分工况

condition_list_6Mpa50Hz = []
condition_list_6Mpa40Hz = []
condition_list_6Mpa30Hz = []
condition_list_6Mpa20Hz = []
condition_list_6Mpa15Hz = []
condition_list_593Mpa50Hz = []
condition_list_589Mpa50Hz = []
condition_list_58Mpa50Hz = []
condition_list_5562Mpa50Hz = []
condition_list_5Mpa50Hz = []
condition_list_5Mpa30Hz = []
condition_list_4Mpa50Hz = []
condition_list_3Mpa50Hz = []
condition_list_355Mpa50Hz = []
condition_list_23Mpa30Hz = []
condition_list_2Mpa50Hz = []
condition_list_1Mpa50Hz = []
condition_list_0Mpa50Hz = []

for i in all_data:

    if (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).all():
        condition_list_6Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]).all():
        condition_list_6Mpa40Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]).all():
        condition_list_6Mpa30Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]).all():
        condition_list_6Mpa20Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]).all():
        condition_list_6Mpa15Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all():
        condition_list_593Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_589Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_58Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_5562Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_5Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():
        condition_list_5Mpa30Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_4Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_3Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_355Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():
        condition_list_23Mpa30Hz.append(i)
    elif (i[2063:2081] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_2Mpa50Hz.append(i)
    elif (i[2063:2081] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_1Mpa50Hz.append(i)
    elif (i[2063:2081] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
        condition_list_0Mpa50Hz.append(i)

condition_list_6Mpa50Hz = np.array(condition_list_6Mpa50Hz)
condition_list_6Mpa40Hz = np.array(condition_list_6Mpa40Hz)
condition_list_6Mpa30Hz = np.array(condition_list_6Mpa30Hz)
condition_list_6Mpa20Hz = np.array(condition_list_6Mpa20Hz)
condition_list_6Mpa15Hz = np.array(condition_list_6Mpa15Hz)
condition_list_593Mpa50Hz = np.array(condition_list_593Mpa50Hz)
condition_list_589Mpa50Hz = np.array(condition_list_589Mpa50Hz)
condition_list_58Mpa50Hz = np.array(condition_list_58Mpa50Hz)
condition_list_5562Mpa50Hz = np.array(condition_list_5562Mpa50Hz)
condition_list_5Mpa50Hz = np.array(condition_list_5Mpa50Hz)
condition_list_5Mpa30Hz = np.array(condition_list_5Mpa30Hz)
condition_list_4Mpa50Hz = np.array(condition_list_4Mpa50Hz)
condition_list_3Mpa50Hz = np.array(condition_list_3Mpa50Hz)
condition_list_355Mpa50Hz = np.array(condition_list_355Mpa50Hz)
condition_list_23Mpa30Hz = np.array(condition_list_23Mpa30Hz)
condition_list_2Mpa50Hz = np.array(condition_list_2Mpa50Hz)
condition_list_1Mpa50Hz = np.array(condition_list_1Mpa50Hz)
condition_list_0Mpa50Hz = np.array(condition_list_0Mpa50Hz)

"""
condition_list_all = [
    condition_list_6Mpa50Hz,
    condition_list_6Mpa40Hz,
    condition_list_6Mpa30Hz,
    condition_list_6Mpa20Hz,
    condition_list_6Mpa15Hz,
    condition_list_593Mpa50Hz,
    condition_list_589Mpa50Hz,
    condition_list_58Mpa50Hz,
    condition_list_5562Mpa50Hz,
    condition_list_5Mpa50Hz,
    condition_list_5Mpa30Hz,
    condition_list_4Mpa50Hz,
    condition_list_3Mpa50Hz,
    condition_list_355Mpa50Hz,
    condition_list_23Mpa30Hz,
    condition_list_2Mpa50Hz,
    condition_list_1Mpa50Hz,
    condition_list_0Mpa50Hz
]
"""


# for i in condition_list_all:
#     print(i.shape)

# 2.工况分离结束后再针对每个工况下的数据分传感器

def find_sensor(condition_list):

    sensor1 = []
    sensor2 = []
    sensor3 = []
    sensor4 = []
    sensor5 = []
    sensor6 = []
    sensor7 = []
    sensor8 = []
    sensor9 = []
    sensor10 = []
    sensor11 = []
    sensor12 = []
    sensor13 = []
    sensor14 = []
    sensor15 = []

    for i in condition_list:
        
        if (i[2048:2063] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor1.append(i)
        elif (i[2048:2063] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor2.append(i)
        elif (i[2048:2063] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor3.append(i)
        elif (i[2048:2063] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor4.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor5.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor6.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor7.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all():
            sensor8.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all():
            sensor9.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all():
            sensor10.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all():
            sensor11.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all():
            sensor12.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():
            sensor13.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).all():
            sensor14.append(i)
        elif (i[2048:2063] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
            sensor15.append(i)
    
    sensor1 = np.array(sensor1)
    sensor2 = np.array(sensor2)
    sensor3 = np.array(sensor3)
    sensor4 = np.array(sensor4)
    sensor5 = np.array(sensor5)
    sensor6 = np.array(sensor6)
    sensor7 = np.array(sensor7)
    sensor8 = np.array(sensor8)
    sensor9 = np.array(sensor9)
    sensor10 = np.array(sensor10)
    sensor11 = np.array(sensor11)
    sensor12 = np.array(sensor12)
    sensor13 = np.array(sensor13)
    sensor14 = np.array(sensor14)
    sensor15 = np.array(sensor15)

    sensor_list_all = [
        sensor1,
        sensor2,
        sensor3,
        sensor4,
        sensor5,
        sensor6,
        sensor7,
        sensor8,
        sensor9,
        sensor10,
        sensor11,
        sensor12,
        sensor13,
        sensor14,
        sensor15
    ]

    return sensor_list_all
       
# for i in find_sensor(condition_list_6Mpa50Hz):
#     print(i.shape)

#-----------------------------计算时域特征值函数---------------------------------
# calcu_feas函数
# （目的）计算时域特征值
def calcu_feas(vibration):
    # 峰值，均值，均方根
    # 峭度，偏度，裕度，波形，脉冲因子，峰值因子
    # 峰值
    vibra_peak = np.array([np.max(abs(vibration[i])) for i in range(len(vibration))])
    # 均值
    vibra_mean = np.array([np.mean(vibration[i]) for i in range(len(vibration))])
    # 均方根值
    vibra_rms = np.array([np.sqrt(np.mean(pow(vibration[i], 2))) for i in range(len(vibration))])

    # 峭度
    kurtosis = np.array([stats.kurtosis(vibration[i].astype(float)) for i in range(len(vibration))])
    # 裕度因子
    margin_factor = vibra_peak / np.array([(np.mean(np.abs(vibration[i])))**2 for i in range(len(vibration))])
    # 波形因子
    waveform_factor = vibra_rms / np.array([(np.mean(np.abs(vibration[i]))) for i in range(len(vibration))])
    # 脉冲因子
    pulse_factor = vibra_peak / np.array([(np.mean(np.abs(vibration[i]))) for i in range(len(vibration))])
    # 峰值因子
    peak_factor = vibra_peak / vibra_rms


    featuretime_list_time = [vibra_peak,vibra_mean, vibra_rms,
                             kurtosis, margin_factor,
                        waveform_factor, pulse_factor, peak_factor]
    return featuretime_list_time

# ----------------------计算频域特征值函数-----------------------

def nextpow2(x): #辅助fft变换函数，无特殊意义，配合Do_fft使用
    if x == 0:
        return 0 
    else:
        return int(np.ceil(np.log2(x)))

def Do_fft(sig,Fs): #fft变换函数，输入信号和采样频率
    xlen = len(sig)
    sig = sig - sig.mean()
    NFFT = 2**nextpow2(xlen)
    yf = np.fft.fft(sig,NFFT)/xlen*2
    yf = abs(yf[0:int(NFFT/2+1)])
    f = Fs/2*np.linspace(0,1,int(NFFT/2+1))
    f = f[:]
    return f,yf  #返回横纵坐标

def transform(vibration):  # 将所有数据的时域振动幅值变为频域幅值，遍历使用fft
    f_list = []
    yf_list = []
    for i in vibration:
        global f
        global yf
        f, yf = Do_fft(i, 2048)
        f_list.append(f)
        yf_list.append(yf)
    f_list = np.array(f_list).reshape(-1, len(f))
    yf_list = np.array(yf_list).reshape(-1, len(yf))
    return f_list, yf_list  # 返回频率list和幅值list为np.array格式
    
def calcu_feas_spec(f_list, yf_list):
    # 平均频率
    p1 = np.array([np.mean(yf_list[i]) for i in range(len(yf_list))])
    # 中心频率
    p2 = np.array([(sum(yf_list[i]*f_list[i])/sum(yf_list[i]))for i in range(len(yf_list))])
    # 频率均方根
    p3 = np.array([np.sqrt(sum((yf_list[i]-p1[i])**2)/len(yf_list[i]))for i in range(len(yf_list))])
    # 频率标准差
    p_ = np.array([np.sqrt(sum((f_list[i]-p2[i])**2*yf_list[i])/len(yf_list[i]))for i in range(len(yf_list))])
    p4 = np.array([sum(abs(f_list[i]-p2[i])*yf_list[i])/(np.sqrt(p_[i])*len(yf_list[i]))for i in range(len(yf_list))])

    featuretime_list_spec = [p1, p2, p3, p4]
    return featuretime_list_spec

# ------------------------区分16类运行状态---------------------------

def operation_state(sensor_list):
    
    type1 = []
    type2 = []
    type3 = []
    type4 = []
    type5 = []
    type6 = []
    type7 = []
    type8 = []
    type9 = []
    type10 = []
    type11 = []
    type12 = []
    type13 = []
    type14 = []
    type15 = []
    type16 = []

    for i in sensor_list:
        
        if (i[2081:] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type1.append(i)
        elif (i[2081:] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type2.append(i)
        elif (i[2081:] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type3.append(i)
        elif (i[2081:] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type4.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type5.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type6.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type7.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            type8.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all():
            type9.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all():
            type10.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all():
            type11.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all():
            type12.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all():
            type13.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():
            type14.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).all():
            type15.append(i)
        elif (i[2081:] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all():
            type16.append(i)

    type1 = np.array(type1)
    type2 = np.array(type2)
    type3 = np.array(type3)
    type4 = np.array(type4)
    type5 = np.array(type5)
    type6 = np.array(type6)
    type7 = np.array(type7)
    type8 = np.array(type8)
    type9 = np.array(type9)
    type10 = np.array(type10)
    type11 = np.array(type11)
    type12 = np.array(type12)
    type13 = np.array(type13)
    type14 = np.array(type14)
    type15 = np.array(type15)
    type16 = np.array(type16)

    type_list_all = [
        type1,
        type2,
        type3,
        type4,
        type5,
        type6,
        type7,
        type8,
        type9,
        type10,
        type11,
        type12,
        type13,
        type14,
        type15,
        type16
    ]

    return type_list_all

# for i in operation_state(find_sensor(condition_list_6Mpa50Hz)[0]):
#     print(i.shape)

sensor_name_list = ['底座东南', '底座东北', '底座西北', '底座西南',
                    '曲轴轴承', '西柱塞密封函', '中柱塞密封函',
                    '东柱塞密封函', '电机东', '电机西', 
                    '曲轴东', '泵头正上方', '泵头正面', 
                    '泵进口管线', '泵出口管线']

features_name_list = ['峰值','均值','均方根',
                      '峭度','裕度因子','波形因子','脉冲因子','峰值因子',
                      '平均频率','中心频率','频率均方根','频率标准差']

state_name_list = ['轴瓦磨损;泵头取弹簧', '泵头取弹簧', '泵头松动', '轴瓦磨损',
                   '轴瓦磨损;地脚螺栓松动', '轴瓦磨损;电机螺栓松动', '轴瓦磨损;电机偏心',
                   '电机偏心', '十字头磨损', '双轴承支架破坏', '西边轴承滚珠外圈磨损',
                   '双轴承滚珠外圈破坏性磨损', '正常运行', '轴承支架破坏', '柱塞松动',
                   '柱塞中磨损']
# 每张图有16个箱，对应上述state，13号箱为正常运行数据

# ------------------------开始画图---------------------------

sensor_index = 0  # 当前传感器序号

for sensor in find_sensor(condition_list_5562Mpa50Hz):  # 得到该工况下各传感器的数据
    
    list = []  # 调整
    for state in operation_state(sensor): # 得到该传感器对应16个状态下的数据
        if len(state) != 0:  # 判断该状态下数据是否为空
            curr_vibration = state[:, 0:2048]
            f_list, yf_list = transform(curr_vibration) 
            features_list1 = calcu_feas(vibration=curr_vibration)  # 计算时域特征
            features_list2 = calcu_feas_spec(f_list, yf_list)  # 计算频域特征
            features_list = features_list1 + features_list2  # 合并特征
            features_list = np.array(features_list)
            # print(features_list.shape)
        else:
            features_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(12, 1)
            # print(features_list.shape)
        list.append(features_list)

    # list（16状态--12维特征--数据条数）
    # for i in list:
    #     print(i.shape)
    # break
    
    FIG = plt.figure(figsize=(70, 40))
    plt.rcParams.update({'font.size': 30})

    for index in range(12):
        plt.subplot(3,4,index+1)
        list_single = []
        for i in list:
            list_single.append((i[index, :]).reshape(-1))  # 把同一特征挑出来，展开成一条向量，最后list_single一共有16行，每行对应一个状态下的这个特征值的所有数据
        plt.boxplot( 
            list_single,
            medianprops={'color': 'red', 'linewidth': '3'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '3'},
            flierprops={"marker": "o", "markerfacecolor": "red",  "markersize": 10},
            )
        plt.title('{}传感器{}特征'.format(str(sensor_name_list[sensor_index]), str(features_name_list[index])), fontproperties='SimHei')

    plt.tight_layout()

    FIG.savefig('./5562Mpa50Hz/'+'5562Mpa50Hz{}传感器'.format(str(sensor_name_list[sensor_index]), dpi=300))
    
    sensor_index = sensor_index + 1
    