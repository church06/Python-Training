import os.path

import bdpy
import h5py
import matplotlib.pyplot as plt
import numpy as np
from bdpy.preproc import select_top
from bdpy.stats import corrcoef


def data_prepare(subject):
    print('Start plotting...')
    print('-----------------')

    target = 's5'
    data = subject[target]
    switch = 3

    # ----- ROI & image feature plot
    for sbj in subject:
        data = subject[sbj]

        if switch == 0:
            print('Plot %s rois...' % sbj)
            subject_plot(data, sbj)
            print('Finish --------')

        elif switch == 1:
            print('Plot %s image features...' % sbj)
            img_feature_plot(data, sbj, image_feature, layers)
            print('Finish ------------------')

    # ----- Correlation rate & layers plot
    for roi in rois:
        x_roi = data.select(rois[roi])

        if switch == 2:
            print('Plot %s %s correlation parameters...' % (target, roi))
            correlation_parameters_plot(data, x_roi, image_feature, layers, target, roi)
            print('Finish -----------------------------')

        elif switch == 3:
            print('Plot %s %s roi layers...' % (target, roi))
            roi_plot(data, x_roi, target, roi)
            print('Finish -----------------------------')


# Plot functions - only for plot data *************************************
def correlation_parameters_plot(data, x_roi, img_feature, layer_all, sbj, roi):
    print('======================================')
    print('subject: %s, roi: %s' % (sbj, roi))

    plt.figure(figsize=(240, 100))
    plt.rcParams['font.size'] = 60

    i = 0

    for layer in layer_all:
        i += 1

        cor_current = x_layer(data, x_roi, img_feature, layer, 1)

        plt.subplot(5, 3, i)
        plt.title('%s %s %s' % (sbj.capitalize(), roi.capitalize(), layer.capitalize()),
                  fontsize=100, fontstyle='italic', fontweight='medium')
        plt.xlabel('voxel', color='r')
        plt.ylabel('correlation rate', color='r')
        plt.bar(range(cor_current.shape[0]), cor_current)
        plt.grid()

    plt.suptitle('%s Correlation Rates' % sbj, fontsize=200, fontstyle='italic', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show(block=False)
    plt.savefig('plots/%s_layers/cor_parameters/%s_%s_correlation.png' % (sbj, sbj, roi))
    plt.close('all')


def roi_plot(data, x_roi, sbj, roi):
    plt.figure(figsize=(240, 100))
    plt.rcParams['font.size'] = 60
    i = 0

    for layer in layers:
        i += 1
        x_current = x_layer(data, x_roi, image_feature, layer, 0)

        plt.subplot(5, 3, i)
        plt.title('%s %s %s' % (sbj.capitalize(), roi.capitalize(), layer.capitalize()),
                  fontsize=100, fontstyle='italic', fontweight='medium')
        plt.xlabel('voxel', color='r')
        plt.ylabel('mean amplitude', color='r')
        plt.bar(range(x_current.shape[1]), x_current[0, :])
        plt.grid()

    plt.suptitle('%s ROIs' % sbj, fontsize=200, fontstyle='italic', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show(block=False)
    plt.savefig('plots/%s_layers/rois/%s_%s.png' % (sbj, sbj, roi))
    plt.close('all')


def subject_plot(data, sbj: ('s1', 's2', 's3', 's4', 's5')):
    plt.figure(figsize=(180, 100))
    plt.rcParams['font.size'] = 60
    i = 0

    for roi in rois:
        i += 1
        x_current = data.select(rois[roi])

        plt.subplot(5, 2, i)
        plt.title('%s x %s' % (sbj.capitalize(), roi.capitalize()),
                  fontsize=100, fontstyle='italic', fontweight='medium')
        plt.xlabel('voxel', color='r')
        plt.ylabel('mean amplitude', color='r')
        plt.bar(range(x_current.shape[1]), x_current[0, :])
        plt.grid()

    plt.suptitle('Subject %s' % sbj[-1], fontsize=200, fontstyle='italic', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show(block=False)
    plt.savefig('plots/%s_layers/rois/%s_rois.png' % (sbj, sbj))
    plt.close('all')


def img_feature_plot(data, sbj: ('s1', 's2', 's3', 's4', 's5'), img_feature, layer_all):
    labels = data.select('stimulus_id')
    y_label = img_feature.select('ImageID')

    data_type = data.select('DataType')

    i_train = (data_type == 1).flatten()
    i_test_seen = (data_type == 2).flatten()
    i_test_img = (data_type == 3).flatten()

    i = 0
    plot_num = len(layer_all)

    plt.figure(figsize=(240, 220))
    plt.suptitle('Image Feature', fontsize=200, fontstyle='italic', fontweight='bold')
    plt.rcParams['font.size'] = 60

    for layer in layer_all:

        y_current_layer = bdpy.get_refdata(img_feature.select(layer), y_label, labels)
        y_train = y_current_layer[i_train, :]
        y_test_seen = y_current_layer[i_test_seen, :]
        y_test_img = y_current_layer[i_test_img, :]

        # ----------------------------------------------------------------------- Plot part
        for time in range(0, 3):
            i += 1
            plt.subplot(plot_num, 3, i)

            if time == 0:
                plt.bar(range(y_train.shape[1]), y_train[0, :])
                plt.title('%s Training Data' % layer.capitalize(), fontsize=100, fontstyle='italic',
                          fontweight='medium')

            elif time == 1:
                plt.bar(range(y_test_seen.shape[1]), y_test_seen[0, :])
                plt.title('%s Seen Test Data' % layer.capitalize(), fontsize=100, fontstyle='italic',
                          fontweight='medium')

            else:
                plt.bar(range(y_test_img.shape[1]), y_test_img[0, :])
                plt.title('%s Imaginary Test Data' % layer.capitalize(), fontsize=100, fontstyle='italic',
                          fontweight='medium')

            plt.xlabel('Vector Index', color='r')
            plt.ylabel('Value', color='r')
            plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show(block=False)
    plt.savefig('plots/%s_image_feature.png' % sbj.capitalize())
    plt.close('all')


def x_layer(data, roi, img_feature, layer: str, return_type: (0, 1, 2)):
    y = img_feature.select(layer)

    labels = data.select('stimulus_id')
    y_label = img_feature.select('ImageID')
    # stimulus_id -> Image ID -> data
    y_sort = bdpy.get_refdata(y, y_label, labels)

    # Correlation: get the value of correlation parameters
    # between x and y, adn use choose the highest
    correlation = corrcoef(y_sort[:, 0], roi, var='col')

    x, voxel_index = select_top(roi, np.abs(correlation),
                                num=1000, axis=1,
                                verbose=False)

    if return_type == 0:
        return x
    elif return_type == 1:
        return correlation
    elif return_type == 2:
        return y_sort
    elif return_type == 3:
        return


# ==========================================================================================

subjects = {'s1': os.path.abspath('data/Subject1.h5'),
            's2': os.path.abspath('data/Subject2.h5'),
            's3': os.path.abspath('data/Subject3.h5'),
            's4': os.path.abspath('data/Subject4.h5'),
            's5': os.path.abspath('data/Subject5.h5'),
            'imageFeature': os.path.abspath('data/ImageFeatures.h5')}

rois = {'VC': 'ROI_VC = 1', 'LVC': 'ROI_LVC = 1', 'HVC': 'ROI_HVC = 1',
        'V1': 'ROI_V1 = 1', 'V2': 'ROI_V2 = 1', 'V3': 'ROI_V3 = 1', 'V4': 'ROI_V4 = 1',
        'LOC': 'ROI_LOC = 1', 'FFA': 'ROI_FFA = 1', 'PPA': 'ROI_PPA = 1'}

voxel = {'VC': 1000, 'LVC': 1000, 'HVC': 1000,
         'V1': 500, 'V2': 500, 'V3': 500,
         'V4': 500,
         'LOC': 500, 'FFA': 500, 'PPA': 500}

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
print('s1: %s\n''s2: %s\n''s3: %s\n''s4: %s\n''s5: %s' % (dataset['s1'].dataset.shape,
                                                          dataset['s2'].dataset.shape,
                                                          dataset['s3'].dataset.shape,
                                                          dataset['s4'].dataset.shape,
                                                          dataset['s5'].dataset.shape))

# dataset & metadata collected
print('\n=======================================')
print('Analyzing...\n')

data_prepare(dataset)
