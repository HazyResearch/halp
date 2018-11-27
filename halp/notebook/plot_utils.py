import numpy as np
import os, sys
import _pickle as cp
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

def save_csv_with_error_bar(data_list, file_name="./test/test.csv", ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    df_list = []
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
        average_y = average_results_array(y)
        std_y = std_results_array(y)
        if ave_x:
            x = average_results_array(x)
        x = np.array(x)
        if len(x.shape) == 2:
            n_pt = x.shape[1]
            x = np.mean(x, axis=0).reshape((n_pt, ) )
        average_y = np.array(average_y)
        std_y = np.array(std_y)
        assert x.shape == average_y.shape
        assert x.shape == std_y.shape
        df_list.append(pd.DataFrame(np.reshape(x, [x.size, 1] ), columns = [label + "|x" ] ) )
        df_list.append(pd.DataFrame(np.reshape(average_y, [average_y.size, 1] ), columns = [label + "|y" ] ) )
        df_list.append(pd.DataFrame(np.reshape(std_y, [std_y.size, 1] ), columns = [label + "|y_std" ] ) )
    pd.concat(df_list, axis=1).to_csv(file_name)
    
    
def save_csv(data_list, file_name="./test/test.csv", ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    df_list = []
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
        df_list.append(pd.DataFrame(np.reshape(x, [x.size, 1] ), columns = [label + "|x" ] ) )
        df_list.append(pd.DataFrame(np.reshape(y, [y.size, 1] ), columns = [label + "|y" ] ) )
    pd.concat(df_list, axis=1).to_csv(file_name)