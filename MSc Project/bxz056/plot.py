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

    # Model setting:
    model_setting = 1
    # 0: print all ROI for each subject
    # 1: print image feature
    # 2: print correlation parameters for  each hierarchical feature of each subject
    # 3: print fMRI data for each hierarchical feature of each subject

    # ------------------------------------------------------------- ROI & image feature plot
    for sbj in subject:
        data_single = subject[sbj]

        # dataset of subject ROIs
        if model_setting == 0:
            print('Plot %s rois...' % sbj)
            subject_plot(data_single, sbj)
            print('Finish --------')

        # dataset of image feature
        elif model_setting == 1:
            print('Plot image features...')
            img_feature_plot(data_single, image_feature, layers, pattern=1)
            print('Finish ------------------')
            break

    # ----------------------------------------------------------------------------------
    # Correlation rate & layers plot

    # Setting: switch: (2, 3)
    #          target: (s1 -> s5) !! manually, automatically will cause memory error

    target = 's5'
    data = subject[target]

    for roi in rois:
        x_roi = data.select(rois[roi])

        if model_setting == 2:
            print('correlation parameters: [subject - %s, roi - %s]' % (target, roi))
            correlation_parameters_plot(data, x_roi, image_feature, layers, target, roi, voxel[roi], pattern=1)
            print('Finish ----------------------------------------------')

        elif model_setting == 3:
            print('Plot %s %s hierarchical fMRI data...' % (target, roi))
            roi_plot(data, x_roi, target, roi, voxel[roi], pattern=0)
            print('Finish -----------------------------')

    print('All plotting Finished. ヾ(•ω•`)o')


# Plot functions - only for plot data *************************************
def correlation_parameters_plot(data, x_roi, img_feature, layer_all, sbj, roi, voxel_roi, pattern: (0, 1)):
    plt.figure(figsize=(240, 100))
    plt.rcParams['font.size'] = 60

    i = 0

    for layer in layer_all:
        i += 1

        cor_current = x_layer(data, x_roi, img_feature, layer, voxel_roi, 1)

        plt.subplot(5, 3, i)
        plt.title('%s - %s' % (roi.capitalize(), layer.capitalize()),
                  fontsize=100, fontstyle='italic', fontweight='medium')

        # Normal correlation rate
        if pattern == 0:
            plt.bar(range(cor_current.shape[0]), cor_current)

        # Absolute correlation rate
        elif pattern == 1:
            cor_current = np.abs(cor_current)
            plt.bar(range(cor_current.shape[0]), cor_current)

        plt.xlabel('voxel', color='r')
        plt.ylabel('correlation rate', color='r')
        plt.grid()

    if pattern == 0:
        plt.suptitle('%s Correlation Rates' % sbj, fontsize=200, fontstyle='italic', fontweight='bold')
    elif pattern == 1:
        plt.suptitle('%s Correlation Rates - Absolute Value' % sbj, fontsize=200, fontstyle='italic', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    path_norm = 'plots/%s_layers/cor_parameters/norm' % sbj
    path_abs = 'plots/%s_layers/cor_parameters/abs' % sbj

    if not os.path.exists(path_norm):
        print('No plots folder. Folder will be created.')
        os.makedirs(path_norm)

    if not os.path.exists(path_abs):
        print('No plots folder. Folder will be created.')
        os.makedirs(path_abs)
        print('Folder create finished.')

    if pattern == 0:
        plt.savefig('plots/%s_layers/cor_parameters/norm/%s_%s_correlation.png' % (sbj, sbj, roi))
    elif pattern == 1:
        plt.savefig('plots/%s_layers/cor_parameters/abs/%s_%s_correlation_abs.png' % (sbj, sbj, roi))
    plt.close('all')


def roi_plot(data, x_roi, sbj, roi, num_voxel, pattern: (0, 1)):
    plt.figure(figsize=(240, 100))
    plt.rcParams['font.size'] = 60
    i = 0

    for layer in layers:

        if pattern == 0:
            i += 1
            x_current = x_layer(data, x_roi, image_feature, layer, num_voxel, 0)
            print('x_current: ', x_current.shape)

            plt.subplot(5, 3, i)
            plt.title('%s' % layer.capitalize(),
                      fontsize=100, fontstyle='italic', fontweight='medium')
            plt.xlabel('voxel', color='r')
            plt.ylabel('mean amplitude', color='r')
            plt.bar(range(x_current.shape[1]), x_current[0, :])
            plt.grid()

    file_output = 'plots/%s_layers/rois' % sbj

    if not os.path.exists(file_output):
        print('No plots folder. Folder will be created.')
        os.makedirs(file_output)
        print('Folder create finished.')

    plt.suptitle('%s %s' % (sbj.capitalize(), roi.capitalize()), fontsize=200, fontstyle='italic', fontweight='bold')
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
        plt.title('%s' % (roi.capitalize()),
                  fontsize=100, fontstyle='italic', fontweight='medium')
        plt.xlabel('voxel', color='r')
        plt.ylabel('mean amplitude', color='r')
        plt.bar(range(x_current.shape[1]), x_current[0, :])
        plt.grid()

    file_output = 'plots/%s_layers/rois' % sbj

    if not os.path.exists(file_output):
        print('No plots folder. Folder will be created.')
        os.makedirs(file_output)
        print('Folder create finished.')

    plt.suptitle('Subject %s ROIs' % sbj[-1], fontsize=200, fontstyle='italic', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show(block=False)
    plt.savefig('plots/%s_layers/rois/%s_rois.png' % (sbj, sbj))
    plt.close('all')


def img_feature_plot(data, img_feature, layer_all, pattern: (0, 1, 2, 3)):
    labels = data.select('stimulus_id')
    y_label = img_feature.select('ImageID')

    data_type = data.select('DataType')

    i_train = (data_type == 1).flatten()
    i_test_seen = (data_type == 2).flatten()
    i_test_img = (data_type == 3).flatten()

    i_test = i_test_seen + i_test_img

    i = 0
    plot_num = len(layer_all)

    # Size not available for pattern 3
    if pattern != 3:
        plt.figure(figsize=(240, 220))
        plt.rcParams['font.size'] = 60
        plt.suptitle('Image Feature', fontsize=200, fontstyle='italic', fontweight='bold')

    for layer in layer_all:

        y_current_layer = bdpy.get_refdata(img_feature.select(layer), y_label, labels)
        y_train = y_current_layer[i_train, :]
        y_test_seen = y_current_layer[i_test_seen, :]
        y_test_img = y_current_layer[i_test_img, :]

        # =================================================================================
        # ----------------------------------------------------------------------- Plot part
        if pattern == 0:

            for time in range(0, 3):
                i += 1
                plt.subplot(plot_num, 3, i)
                plt.ylim()
                plt.xlabel('Visualwords (SIFT Descriptor)', color='r')
                plt.ylabel('Frequency', color='r')
                plt.grid()

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

        # Combine seen and imaginary chart ------------------------------------------------
        elif pattern == 1:
            y_test = y_current_layer[i_test, :]

            for time in range(0, 2):
                i += 1
                plt.subplot(plot_num, 2, i)
                if layer == 'cnn1' or layer == 'cnn2' or layer == 'cnn3' or \
                        layer == 'cnn4' or layer == 'cnn5' or layer == 'cnn6' or \
                        layer == 'cnn7':
                    plt.xlabel('1000 random units', color='r')
                    plt.ylabel('Feature value', color='r')

                elif layer == 'cnn8':
                    plt.xlabel('All unit', color='r')
                    plt.ylabel('Feature Value', color='r')

                elif layer == 'hmax1':
                    plt.xlabel('S1 1000 random units', color='r')
                    plt.ylabel('Feature Value', color='r')

                elif layer == 'hmax2':
                    plt.xlabel('S2 & C2 1000 random units', color='r')
                    plt.ylabel('C2 Feature', color='r')

                elif layer == 'hmax3':
                    plt.xlabel('C3 all units', color='r')
                    plt.ylabel('C3 Feature', color='r')

                elif layer == 'gist':
                    plt.xlabel('All units', color='r')
                    plt.ylabel('GIST Feature (16 × 4 × 16)', color='r')

                elif layer == 'sift':
                    plt.xlabel('Frequency', color='r')
                    plt.ylabel('SIFT descriptor', color='r')

                plt.grid()

                if time == 0:
                    plt.bar(range(y_train.shape[1]), y_train[0, :])
                    plt.title('%s Training Data' % layer.capitalize(),
                              fontsize=100, fontstyle='italic',
                              fontweight='medium')

                else:
                    plt.bar(range(y_test.shape[1]), y_test[0, :])
                    plt.title('%s Test Data' % layer.capitalize(),
                              fontsize=100, fontstyle='italic',
                              fontweight='medium')

        elif pattern == 2:
            i += 1

            norm_scale_y = np.array([])

            norm_mean_y = np.mean(y_train, axis=0)
            std_y = np.std(y_train, axis=0, ddof=1)

            for std in std_y:
                if std == 0:
                    norm_scale_y = np.append(norm_scale_y, 1)
                else:
                    norm_scale_y = np.append(norm_scale_y, std)

            y_train = (y_train - norm_mean_y) / norm_scale_y

            plt.subplot(plot_num, 1, i)
            plt.xlabel('Visualwords (SIFT Descriptor)', color='r')
            plt.ylabel('Frequency', color='r')
            plt.grid()

            plt.bar(range(y_train.shape[1]), y_train[0, :])
            plt.title('%s Training Data' % layer.capitalize(),
                      fontsize=100, fontstyle='italic',
                      fontweight='medium')

        elif pattern == 3:
            plt.bar(range(y_train.shape[1]), y_train[0, :])
            plt.title('S1 %s Sample' % layer.capitalize(),
                      fontstyle='italic',
                      fontweight='medium')
            break

        # ----------------------------------------------------------------------- Plot part
        # =================================================================================

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show(block=False)

    file_output = 'plots'

    if not os.path.exists(file_output):
        print('No plots folder. Folder will be created.')
        os.makedirs(file_output)
        print('Folder create finished.')

    if pattern == 0:
        plt.savefig('plots/Image_Feature.png')
    elif pattern == 1:
        plt.savefig('plots/IMG_F_Test_all.png')
    elif pattern == 2:
        plt.savefig('plots/Y_train_Z_score.png')

    plt.close('all')


def x_layer(data, roi, img_feature, layer: str, voxel_roi, return_type: (0, 1, 2)):
    y = img_feature.select(layer)

    labels = data.select('stimulus_id')
    y_label = img_feature.select('ImageID')
    # stimulus_id -> Image ID -> data
    y_sort = bdpy.get_refdata(y, y_label, labels)

    # Correlation: get the value of correlation parameters
    # between x and y, adn use choose the highest
    correlation = corrcoef(y_sort[:, 0], roi, var='col')

    x, voxel_index = select_top(roi, np.abs(correlation),
                                num=voxel_roi, axis=1,
                                verbose=False)

    if return_type == 0:
        return np.array(x)
    elif return_type == 1:
        return np.array(correlation)
    elif return_type == 2:
        return np.array(y_sort)


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
data_prepare(dataset)
