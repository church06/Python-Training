import mne
import dataOptimizer as dO
import pandas as pd
import os


def get_data():
    print('=========== Program Starting ===========')

    data_path = 'E://Entrance//Coding//datasets//fNIRs//rob-luke-BIDS-NIRS-Tapping'

    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if '.snirf' in name:
                print(os.path.join(root, name))
