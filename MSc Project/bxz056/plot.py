import os
import os.path
from itertools import product

import bdpy
import h5py
import matplotlib.pyplot as plt
import numpy as np
from bdpy.preproc import select_top
from bdpy.stats import corrcoef


def data_prepare(subject, roi_all, img_feature, layer_all, voxel_all):
    print('Start plotting...')
    print('-----------------')

    for roi in roi_all:
        data = subject['s1']
        x_roi = data.select(roi_all[roi])
        print('x_roi in data_prepare: ', x_roi.shape)

        correlation_parameters_plot(data, x_roi, img_feature, 's1', roi)


def correlation_parameters_plot(data, x_roi, y_layers_all, sbj, roi):
    print('======================================')
    print('subject: %s, roi: %s' % (sbj, roi))

    x_cnn1, cor_cnn1 = x_layer(data, x_roi, y_layers_all, 'cnn1')
    x_cnn2, cor_cnn2 = x_layer(data, x_roi, y_layers_all, 'cnn2')
    x_cnn3, cor_cnn3 = x_layer(data, x_roi, y_layers_all, 'cnn3')
    x_cnn4, cor_cnn4 = x_layer(data, x_roi, y_layers_all, 'cnn4')
    x_cnn5, cor_cnn5 = x_layer(data, x_roi, y_layers_all, 'cnn5')
    x_cnn6, cor_cnn6 = x_layer(data, x_roi, y_layers_all, 'cnn6')
    x_cnn7, cor_cnn7 = x_layer(data, x_roi, y_layers_all, 'cnn7')
    x_cnn8, cor_cnn8 = x_layer(data, x_roi, y_layers_all, 'cnn8')

    x_hmax1, cor_hmax1 = x_layer(data, x_roi, y_layers_all, 'hmax1')
    x_hmax2, cor_hmax2 = x_layer(data, x_roi, y_layers_all, 'hmax2')
    x_hmax3, cor_hmax3 = x_layer(data, x_roi, y_layers_all, 'hmax3')

    x_gist, cor_gist = x_layer(data, x_roi, y_layers_all, 'gist')
    x_sift, cor_sift = x_layer(data, x_roi, y_layers_all, 'sift')

    # print('cor_cnn1: ', cor_cnn1.shape)
    # print('cor_cnn2: ', cor_cnn2.shape)
    # print('cor_cnn3: ', cor_cnn3.shape)
    # print('cor_cnn4: ', cor_cnn4.shape)
    # print('cor_cnn5: ', cor_cnn5.shape)
    # print('cor_cnn6: ', cor_cnn6.shape)
    # print('cor_cnn7: ', cor_cnn7.shape)
    # print('cor_cnn8: ', cor_cnn8.shape)
    # print('cor_hmax1: ', cor_hmax1.shape)
    # print('cor_hmax2: ', cor_hmax2.shape)
    # print('cor_hmax3: ', cor_hmax3.shape)
    # print('cor_gist: ', cor_gist.shape)
    # print('cor_sift: ', cor_sift.shape)


def layers_plot(data, x_roi, y_layers_all, sbj, roi):
    x_cnn1, cor_cnn1 = x_layer(data, x_roi, y_layers_all, 'cnn1')
    x_cnn2, cor_cnn2 = x_layer(data, x_roi, y_layers_all, 'cnn2')
    x_cnn3, cor_cnn3 = x_layer(data, x_roi, y_layers_all, 'cnn3')
    x_cnn4, cor_cnn4 = x_layer(data, x_roi, y_layers_all, 'cnn4')
    x_cnn5, cor_cnn5 = x_layer(data, x_roi, y_layers_all, 'cnn5')
    x_cnn6, cor_cnn6 = x_layer(data, x_roi, y_layers_all, 'cnn6')
    x_cnn7, cor_cnn7 = x_layer(data, x_roi, y_layers_all, 'cnn7')
    x_cnn8, cor_cnn8 = x_layer(data, x_roi, y_layers_all, 'cnn8')

    x_hmax1, cor_hmax1 = x_layer(data, x_roi, y_layers_all, 'hmax1')
    x_hmax2, cor_hmax2 = x_layer(data, x_roi, y_layers_all, 'hmax2')
    x_hmax3, cor_hmax3 = x_layer(data, x_roi, y_layers_all, 'hmax3')

    x_gist, cor_gist = x_layer(data, x_roi, y_layers_all, 'gist')
    x_sift, cor_sift = x_layer(data, x_roi, y_layers_all, 'sift')

    plt.figure(figsize=(240, 100))
    plt.rcParams['font.size'] = 60

    plt.subplot(5, 3, 1)
    plt.title('%s_%s_cnn1' % (sbj, roi))
    plt.bar(range(x_cnn1.shape[1]), x_cnn1[0, :])

    plt.subplot(5, 3, 2)
    plt.title('%s_%s_cnn2' % (sbj, roi))
    plt.bar(range(x_cnn2.shape[1]), x_cnn2[0, :])

    plt.subplot(5, 3, 3)
    plt.title('%s_%s_cnn3' % (sbj, roi))
    plt.bar(range(x_cnn3.shape[1]), x_cnn3[0, :])

    plt.subplot(5, 3, 4)
    plt.title('%s_%s_cnn4' % (sbj, roi))
    plt.bar(range(x_cnn4.shape[1]), x_cnn4[0, :])

    plt.subplot(5, 3, 5)
    plt.title('%s_%s_cnn5' % (sbj, roi))
    plt.bar(range(x_cnn5.shape[1]), x_cnn5[0, :])

    plt.subplot(5, 3, 6)
    plt.title('%s_%s_cnn6' % (sbj, roi))
    plt.bar(range(x_cnn6.shape[1]), x_cnn6[0, :])

    plt.subplot(5, 3, 7)
    plt.title('%s_%s_cnn7' % (sbj, roi))
    plt.bar(range(x_cnn7.shape[1]), x_cnn7[0, :])

    plt.subplot(5, 3, 8)
    plt.title('%s_%s_cnn8' % (sbj, roi))
    plt.bar(range(x_cnn8.shape[1]), x_cnn8[0, :])

    plt.subplot(5, 3, 9)
    plt.title('%s_%s_hmax1' % (sbj, roi))
    plt.bar(range(x_hmax1.shape[1]), x_hmax1[0, :])

    plt.subplot(5, 3, 10)
    plt.title('%s_%s_hmax2' % (sbj, roi))
    plt.bar(range(x_hmax2.shape[1]), x_hmax2[0, :])

    plt.subplot(5, 3, 11)
    plt.title('%s_%s_hmax3' % (sbj, roi))
    plt.bar(range(x_hmax3.shape[1]), x_hmax3[0, :])

    plt.subplot(5, 3, 12)
    plt.title('%s_%s_gist' % (sbj, roi))
    plt.bar(range(x_gist.shape[1]), x_gist[0, :])

    plt.subplot(5, 3, 13)
    plt.title('%s_%s_sift' % (sbj, roi))
    plt.bar(range(x_sift.shape[1]), x_sift[0, :])

    plt.tight_layout()
    plt.savefig('plots/%s_%s_layers.png' % (sbj, roi))
    plt.close('all')


def x_layer(data, x_roi, y_layers_all, layer: str):
    y = y_layers_all.select(layer)

    labels = data.select('stimulus_id')
    y_label = y_layers_all.select('ImageID')
    # stimulus_id -> Image ID -> data
    y_sort = bdpy.get_refdata(y, y_label, labels)

    # Correlation: get the value of correlation parameters
    # between x and y, adn use choose the highest
    correlation = corrcoef(y_sort[:, 0], x_roi, var='col')
    print('x_layer cor: ', correlation.shape)
    print()

    x, voxel_index = select_top(x_roi, np.abs(correlation),
                                num=1000, axis=1,
                                verbose=False)

    return x, correlation


def rois_plot(data, subject, roi_all):
    plt.figure(figsize=(160, 100))
    plt.rcParams['font.size'] = 60

    x_vc = data.select(roi_all['VC'])
    x_lvc = data.select(roi_all['LVC'])
    x_hvc = data.select(roi_all['HVC'])
    x_v1 = data.select(roi_all['V1'])
    x_v2 = data.select(roi_all['V2'])
    x_v3 = data.select(roi_all['V3'])
    x_v4 = data.select(roi_all['V4'])
    x_loc = data.select(roi_all['LOC'])
    x_ffa = data.select(roi_all['FFA'])
    x_ppa = data.select(roi_all['PPA'])

    plt.subplot(5, 2, 1)
    plt.title('%s_x_VC' % subject)
    plt.bar(range(x_vc.shape[1]), x_vc[0, :])

    plt.subplot(5, 2, 2)
    plt.title('%s_x_LVC' % subject)
    plt.bar(range(x_lvc.shape[1]), x_lvc[0, :])

    plt.subplot(5, 2, 3)
    plt.title('%s_x_HVC' % subject)
    plt.bar(range(x_hvc.shape[1]), x_hvc[0, :])

    plt.subplot(5, 2, 4)
    plt.title('%s_x_V1' % subject)
    plt.bar(range(x_v1.shape[1]), x_v1[0, :])

    plt.subplot(5, 2, 5)
    plt.title('%s_x_V2' % subject)
    plt.bar(range(x_v2.shape[1]), x_v2[0, :])

    plt.subplot(5, 2, 6)
    plt.title('%s_x_V3' % subject)
    plt.bar(range(x_v3.shape[1]), x_v3[0, :])

    plt.subplot(5, 2, 7)
    plt.title('%s_x_V4' % subject)
    plt.bar(range(x_v4.shape[1]), x_v4[0, :])

    plt.subplot(5, 2, 8)
    plt.title('%s_x_LOC' % subject)
    plt.bar(range(x_loc.shape[1]), x_loc[0, :])

    plt.subplot(5, 2, 9)
    plt.title('%s_x_FFA' % subject)
    plt.bar(range(x_ffa.shape[1]), x_ffa[0, :])

    plt.subplot(5, 2, 10)
    plt.title('%s_x_PPA' % subject)
    plt.bar(range(x_ppa.shape[1]), x_ppa[0, :])

    plt.tight_layout()
    plt.savefig('plots/%s_rois.png' % subject)


# =================================================

subjects = {'s1': os.path.abspath('data/Subject1.h5'),
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

layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4',
          'cnn5', 'cnn6', 'cnn7', 'cnn8',
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

print('--------------------- data ')
print('s1: %s\n'
      's2: %s\n'
      's3: %s\n'
      's4: %s\n'
      's5: %s' % (dataset['s1'].dataset.shape,
                  dataset['s2'].dataset.shape,
                  dataset['s3'].dataset.shape,
                  dataset['s4'].dataset.shape,
                  dataset['s5'].dataset.shape))

# dataset & metadata collected

print('\n=======================================')
print('Analyzing...\n')

data_prepare(dataset, rois, image_feature, layers, voxel)
