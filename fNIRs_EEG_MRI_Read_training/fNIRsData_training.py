import mne
import dataOptimizer as dO
import pandas as pd
import os


# functions =====================================
def get_data():
    data_all = {}
    print('\n=========== Program Starting ===========')

    data_path = 'E://Entrance//Coding//datasets//fNIRs//rob-luke-BIDS-NIRS-Tapping'

    print('Locate: ', data_path)

    for root, dirs, files in os.walk(data_path, topdown=False):

        for name in files:
            if '.snirf' in name:
                sub = int(name[5:6])

                print('Read File: {0} | Name: {1}'.format(name, sub))

                sub_data = mne.io.read_raw_snirf(os.path.join(root, name))
                print(sub_data)

    return data_all


# main code =====================================
dataset = get_data()
