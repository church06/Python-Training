import os
import os.path
from itertools import product

import bdpy
import h5py
import matplotlib.pyplot as plt
import numpy as np
from bdpy.preproc import select_top
from bdpy.stats import corrcoef


def data_prepare(subject, rois, img_feature, layers, voxel):
    print('Start learning:')
    print('-----------------')

    data = subject['s1_layers']

    for sbj in subject:
        data = subject[sbj]

        plt.figure(figsize=(160, 100))
        plt.rcParams['font.size'] = 60

        x_vc = data.select(rois['VC'])
        x_lvc = data.select(rois['LVC'])
        x_hvc = data.select(rois['HVC'])
        x_v1 = data.select(rois['V1'])
        x_v2 = data.select(rois['V2'])
        x_v3 = data.select(rois['V3'])
        x_v4 = data.select(rois['V4'])
        x_loc = data.select(rois['LOC'])
        x_ffa = data.select(rois['FFA'])
        x_ppa = data.select(rois['PPA'])

        plt.subplot(5, 2, 1)
        plt.title('s1_x_VC')
        plt.bar(range(x_vc.shape[1]), x_vc[0, :])

        plt.subplot(5, 2, 2)
        plt.title('s1_x_LVC')
        plt.bar(range(x_lvc.shape[1]), x_lvc[0, :])

        plt.subplot(5, 2, 3)
        plt.title('s1_x_HVC')
        plt.bar(range(x_hvc.shape[1]), x_hvc[0, :])

        plt.subplot(5, 2, 4)
        plt.title('s1_x_V1')
        plt.bar(range(x_v1.shape[1]), x_v1[0, :])

        plt.subplot(5, 2, 5)
        plt.title('s1_x_V2')
        plt.bar(range(x_v2.shape[1]), x_v2[0, :])

        plt.subplot(5, 2, 6)
        plt.title('s1_x_V3')
        plt.bar(range(x_v3.shape[1]), x_v3[0, :])

        plt.subplot(5, 2, 7)
        plt.title('s1_x_V4')
        plt.bar(range(x_v4.shape[1]), x_v4[0, :])

        plt.subplot(5, 2, 8)
        plt.title('s1_x_LOC')
        plt.bar(range(x_loc.shape[1]), x_loc[0, :])

        plt.subplot(5, 2, 9)
        plt.title('s1_x_FFA')
        plt.bar(range(x_ffa.shape[1]), x_ffa[0, :])

        plt.subplot(5, 2, 10)
        plt.title('s1_x_PPA')
        plt.bar(range(x_ppa.shape[1]), x_ppa[0, :])

        plt.tight_layout()
        plt.savefig('plots/%s_rois.png' % sbj)

# =======================================================================================================

subjects = {'s1_layers': os.path.abspath('data/Subject1.h5'),
            's2': os.path.abspath('data/Subject2.h5'),
            's3': os.path.abspath('data/Subject3.h5'),
            's4': os.path.abspath('data/Subject4.h5'),
            's5': os.path.abspath('data/Subject5.h5'),
            'imageFeature': os.path.abspath('data/ImageFeatures.h5')}

rois = {'VC': 'ROI_VC = 1',
                      'LVC': 'ROI_LVC = 1',
                      'HVC': 'ROI_HVC = 1',
                      'V1': 'ROI_V1 = 1',
                      'V2': 'ROI_V2 = 1',
                      'V3': 'ROI_V3 = 1',
                      'V4': 'ROI_V4 = 1',
                      'LOC': 'ROI_LOC = 1',
                      'FFA': 'ROI_FFA = 1',
                      'PPA': 'ROI_PPA = 1'}

voxel = {'VC': 1000,
         'LVC': 1000,
         'HVC': 1000,
         'V1': 500,
         'V2': 500,
         'V3': 500,
         'V4': 500,
         'LOC': 500,
         'FFA': 500,
         'PPA': 500}

layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
          'hmax1', 'hmax2', 'hmax3',
          'gist', 'sift']

print('=======================================')
print('Data loading...')

dataset = {}
image_feature = {}

file = None

for person in subjects:

    file = h5py.File(subjects[person], 'r')

    if person != 'imageFeature':
        # Subject 1 ~ 5

        print(person, '---------------------')
        print('data: ', file.keys())

        dataset[person] = bdpy.BData(subjects[person])

    else:
        image_feature = bdpy.BData(subjects[person])

file.close()

# dataset & metadata collected

print('\n=======================================')
print('Analyzing...\n')

data_prepare(dataset, rois, image_feature, layers, voxel)
