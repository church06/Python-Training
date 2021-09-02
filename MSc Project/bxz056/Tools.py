import os
from itertools import product

import bdpy
import h5py
import numpy
from bdpy import get_refdata
from bdpy.preproc import select_top
from bdpy.stats import corrcoef


def read_folder(folder: h5py.File, mode: str):
    keys = list(folder.keys())
    output = {}

    if mode == 'group':
        for k in keys:
            output[k] = folder[k]

    elif mode == 'dataset':
        for k in keys:
            output[k] = numpy.array(folder[k])

    return output


def xy_folder_creation(file, sbj_num, roi: str, layer: str, data: dict):
    try:
        s_file = file.create_group(sbj_num)
    except ValueError:
        s_file = file[sbj_num]

    try:
        r_file = s_file.create_group(roi)
    except ValueError:
        r_file = s_file[roi]

    try:
        l_file = r_file.create_group(layer)
    except ValueError:
        l_file = r_file[layer]

    for lab in data:
        try:
            l_file.create_dataset(lab, data=numpy.array(data[lab]))

        except RuntimeError:
            del l_file[lab]
            l_file.create_dataset(lab, data=numpy.array(data[lab]))


class Tool:
    def __init__(self):
        self.data_path = 'data\\'
        self.result_path = 'HDF5s\\Result.hdf5'
        self.XY_T_STD_path = 'HDF5s\\xy_train&std_data.hdf5'
        self.MResult_path = 'HDF5s\\mergedResult.hdf5'
        self.hdf5_path = 'HDF5s\\'
        self.cat_data_path = 'HDF5s\\cat_data.hdf5'
        self.final_data_path = 'HDF5s\\correct_rates.hdf5'

    def read_subject_1(self):
        print('Read subject data')

        folder_dir = self.data_path

        path = {'s1': os.path.abspath(folder_dir + 'Subject1.h5'),
                'imageFeature': os.path.abspath(folder_dir + 'ImageFeatures.h5')}

        s1 = bdpy.BData(path['s1'])
        img = bdpy.BData(path['imageFeature'])

        return s1, img

    def read_xy_std_data(self, sbj_num: str):
        path = self.XY_T_STD_path
        file = h5py.File(path, 'r')

        group = file[sbj_num]

        output = {}

        r_keys = list(group.keys())

        for R in r_keys:
            l_keys = list(group[R].keys())
            L_dict = {}

            for L in l_keys:
                L_dict = read_folder(group[R][L], mode='dataset')

            output[R] = L_dict

        file.close()
        return output

    def read_result_data(self, sbj_num: str):
        print('Getting result File...')

        file = h5py.File(self.result_path, 'r')
        sbj = file[sbj_num]

        r_list = list(sbj.keys())

        R_dict = {}

        for R in r_list:
            l_list = list(sbj[R].keys())
            L_dict = {}

            for F in l_list:
                n_list = list(sbj[R][F].keys())
                N_dict = {}

                for N in n_list:
                    d_list = list(sbj[R][F][N].keys())
                    U_dict = {}
                    D_dict = {}

                    for D in d_list:
                        if D != 'alpha' and D != 'weight' and D != 'gain':
                            D_dict[D] = numpy.array(sbj[R][F][N][D])

                        else:
                            for i in range(0, 200):
                                U_dict[str(i)] = numpy.array(sbj[R][F][N][D][str(i)])

                            D_dict[D] = U_dict
                        N_dict[N] = D_dict
                    L_dict[F] = N_dict
            R_dict[R] = L_dict

        file.close()
        return R_dict

    def read_merged_data(self, sbj_num):
        print('Getting mergedResult File...')

        path = self.MResult_path
        file = h5py.File(path, 'r')
        loc = file[sbj_num]

        r_list = list(loc.keys())

        r_dict = {}

        for R in r_list:
            l_list = list(loc[R].keys())
            l_dict = {}

            for F in l_list:
                n_list = list(loc[R][F].keys())
                n_dict = {}

                for N in n_list:
                    d_list = list(loc[R][F][N].keys())
                    d_dict = {}

                    for D in d_list:
                        d_dict[D] = numpy.array(loc[R][F][N][D])

                    n_dict[N] = d_dict
                l_dict[F] = n_dict
            r_dict[R] = l_dict

        file.close()
        return r_dict

    def read_cat(self, sbj_num: str):
        print('Getting categories.')
        file = h5py.File(self.cat_data_path, 'r')
        target = file[sbj_num]

        keys = ['cat_pt_av', 'cat_im_av', 'cat_pt_label', 'cat_im_label']

        output = {}

        for key in keys:
            output[key] = numpy.array(target[key])

        file.close()
        return output

    def create_xy_std_data(self, sbj_num: str, roi: str, layer: str):
        print("Creating ['%s'] test data by normalization: [none, z-score, min-max, decimal]" % layer)

        file = h5py.File(self.XY_T_STD_path, 'a')
        roi_up = roi.upper()

        voxel = {'VC': 1000, 'LVC': 1000, 'HVC': 1000,
                 'V1': 500, 'V2': 500, 'V3': 500,
                 'V4': 500,
                 'LOC': 500, 'FFA': 500, 'PPA': 500}

        roi_s = {'VC': 'ROI_VC = 1', 'LVC': 'ROI_LVC = 1', 'HVC': 'ROI_HVC = 1',
                 'V1': 'ROI_V1 = 1', 'V2': 'ROI_V2 = 1', 'V3': 'ROI_V3 = 1',
                 'V4': 'ROI_V4 = 1',
                 'LOC': 'ROI_LOC = 1', 'FFA': 'ROI_FFA = 1', 'PPA': 'ROI_PPA = 1'}

        # Labels -----------------------------
        s1, img = self.read_subject_1()
        labels = s1.select('stimulus_id')
        data_type = s1.select('DataType')

        i_train = (data_type == 1).flatten()

        i_test_pt = (data_type == 2).flatten()
        i_test_im = (data_type == 3).flatten()
        i_test = i_test_im + i_test_pt
        # ------------------------------------

        # Image Feature -----------------------------
        y = img.select(layer)
        y_label = img.select('ImageID')
        y_sort = bdpy.get_refdata(y, y_label, labels)

        y_train = y_sort[i_train, :]
        y_train_unit = y_train[:, 0]
        # -------------------------------------------

        # fMRI data -----------------------------------------------------------------------------------
        x = s1.select(roi_s[roi_up])
        x_train = x[i_train, :]
        x_test = x[i_test, :]
        print('x_test: ', x_test.shape)
        correlation = corrcoef(y_train_unit, x_train, var='col')

        print('correlated...')
        x_train, voxel_index = select_top(x_train, numpy.abs(correlation), voxel[roi_up], axis=1, verbose=False)
        x_test = x_test[:, voxel_index]
        print('x_test: ', x_test.shape)
        # ---------------------------------------------------------------------------------------------

        print()

        # No Normalization ------
        n_test = x_test
        n_test_std = numpy.std(x_test)

        n_train = x_train
        n_train_std = numpy.std(x_train)

        n_y = y_train_unit
        n_y_std = numpy.std(n_y)

        print('n_test: ', n_test.shape)
        print('n_y: ', n_y.shape)
        # -----------------------

        # Z-Score -----------------------------------------
        norm_mean_x = numpy.mean(x_train, axis=0)
        norm_scale_x = numpy.std(x_train, axis=0, ddof=1)

        norm_mean_y = numpy.mean(y_train_unit, axis=0)
        std_y = numpy.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        z_test = (x_test - norm_mean_x) / norm_scale_x
        z_test_std = numpy.std(z_test)

        z_train = (x_train - norm_mean_x) / norm_scale_x
        z_train_std = numpy.std(z_train)

        z_y = (y_train_unit - norm_mean_y) / norm_scale_y
        z_y_std = numpy.std(z_y)

        print('z_test: ', z_test.shape)
        print('z_y: ', z_y.shape)
        # -------------------------------------------------

        # Min-Max -----------------------------------------
        x_min = numpy.min(x_train)
        x_max = numpy.max(x_train)

        y_min = numpy.min(y_train_unit)
        y_max = numpy.max(y_train_unit)

        m_test = (x_test - x_min) / (x_max - x_min) * 2 - 1
        m_test_std = numpy.std(m_test)

        m_train = (x_train - x_min) / (x_max - x_min) * 2 - 1
        m_train_std = numpy.std(m_train)

        m_y = (y_train_unit - y_min) / (y_max - y_min) * 2 - 1
        m_y_std = numpy.std(m_y)

        print('m_test: ', m_test.shape)
        print('m_y: ', m_y.shape)
        # -------------------------------------------------

        # Decimal Scaling ----------------------
        x_train_abs = numpy.abs(x_train)
        x_abs_max = numpy.max(x_train_abs)

        power = 1
        while x_abs_max > 1:
            x_abs_max /= 10
            power += 1

        y_train_unit_abs = numpy.abs(y_train_unit)
        y_abs_max = numpy.max(y_train_unit_abs)

        power_y = 1
        while y_abs_max > 1:
            y_abs_max /= 10
            power_y += 1

        d_test = x_test / numpy.power(10, power)
        d_test_std = numpy.std(d_test)

        d_train = x_train / numpy.power(10, power)
        d_train_std = numpy.std(d_train)

        d_y = y_train_unit / numpy.power(10, power_y)
        d_y_std = numpy.std(d_y)

        print('d_test: ', d_test.shape)
        print('d_y: ', d_y.shape)
        # --------------------------------------

        labels = {'n_train': n_train, 'z_train': z_train, 'm_train': m_train, 'd_train': d_train,
                  'n_train_std': n_train_std, 'z_train_std': z_train_std,
                  'm_train_std': m_train_std, 'd_train_std': d_train_std,
                  'n_test': n_test, 'z_test': z_test, 'm_test': m_test, 'd_test': d_test,
                  'n_test_std': n_test_std, 'z_test_std': z_test_std,
                  'm_test_std': m_test_std, 'd_test_std': d_test_std,
                  'n_y': n_y, 'z_y': z_y, 'm_y': m_y, 'd_y': d_y,
                  'n_y_std': n_y_std, 'z_y_std': z_y_std, 'm_y_std': m_y_std, 'd_y_std': d_y_std}

        xy_folder_creation(file, sbj_num, roi_up, layer, labels)
        file.close()

    def create_cat(self, sbj_num: str):
        print('Creating category data')
        path = self.hdf5_path

        s_list = {'s1': 'Subject1.h5', 's2': 'Subject2.h5',
                  's3': 'Subject3.h5', 's4': 'Subject4.h5',
                  's5': 'Subject5.h5'}

        file = h5py.File(path + 'cat_data.hdf5', 'a')
        try:
            out_file = file.create_group(sbj_num)
        except ValueError:
            out_file = file[sbj_num]
        groups = ['cat_pt_av', 'cat_im_av', 'cat_pt_label', 'cat_im_label']

        # Getting data ---------------------------------------
        subject = bdpy.BData(self.data_path + s_list[sbj_num])
        img_feature = bdpy.BData(self.data_path + 'ImageFeatures.h5')
        # ----------------------------------------------------

        layer_feature = img_feature.select('cnn1')

        data_type = subject.select('DataType')
        labels = subject.select('stimulus_id')

        i_test_seen = (data_type == 2).flatten()
        i_test_im = (data_type == 3).flatten()

        # ----------------------------------------------------------------------------------------------------------
        # Copy from
        # https://github.com/KamitaniLab/GenericObjectDecoding/blob/master/code/python/analysis_FeaturePrediction.py
        # Do some refactor

        # Get averaged predicted feature
        test_label_pt = labels[i_test_seen, :].flatten()
        test_label_im = labels[i_test_im, :].flatten()

        # Get category averaged features
        cat_labels_pt = numpy.vstack([int(pt) for pt in test_label_pt])  # Category labels (perception test)
        cat_labels_im = numpy.vstack([int(im) for im in test_label_im])  # Category labels (perception test)

        cat_pt_label = numpy.unique(cat_labels_pt)
        cat_im_label = numpy.unique(cat_labels_im)

        cat_labels = img_feature.select('CatID')  # Category labels in image features
        ind_cat_av = (img_feature.select('FeatureType') == 3).flatten()

        c_pt_av = get_refdata(layer_feature[ind_cat_av, :], cat_labels[ind_cat_av, :], cat_pt_label)
        c_im_av = get_refdata(layer_feature[ind_cat_av, :], cat_labels[ind_cat_av, :], cat_im_label)
        # ----------------------------------------------------------------------------------------------------------

        temp_dict = {groups[0]: c_pt_av, groups[1]: c_im_av,
                     groups[2]: cat_pt_label, groups[3]: cat_im_label}

        for name in groups:
            try:
                print("Create ['%s']." % name)
                out_file.create_dataset(name, data=temp_dict[name])
            except RuntimeError:
                print("['%s'] already exists. Remove file and get new data" % name)
                del out_file[name]
                out_file.create_dataset(name, data=temp_dict[name])

        file.close()

    def save_final_data(self, sbj_num, data: dict):
        file = h5py.File(self.final_data_path, 'a')
        try:
            s_group = file.create_group(sbj_num)
        except ValueError:
            s_group = file[sbj_num]

        for R in data.keys():
            l_list = list(data[R].keys())

            try:
                r_group = s_group.create_group(R)
            except ValueError:
                r_group = s_group[R]

            for L in l_list:
                d_list = list(data[R][L].keys())

                try:
                    l_group = r_group.create_group(L)
                except ValueError:
                    l_group = r_group[L]

                for D in d_list:
                    try:
                        l_group.create_dataset(D, data=data[R][L][D])

                    except RuntimeError:
                        del l_group[D]
                        l_group.create_dataset(D, data=data[R][L][D])

        print('Create [ Correct Ratio file ] Finished.')

    def save_to_result(self, S: str, R: str, L, N: int, data_dict: dict):
        print('Save to Results.hdf5 !!!!!!')
        hdf5_dir = self.result_path
        # r:        Read only
        # r+:       Read / write, file must exist
        # w:        Create file, truncate if exists
        # w- / x:   Create file, fail if exists
        # a:        Read/write if exists, create otherwise
        dataTypes = list(data_dict.keys())
        norm_tec = ['none', 'z-score', 'min-max', 'decimal']

        hdf5 = h5py.File(hdf5_dir, 'a')

        try:
            s_group = hdf5.create_group(S)
        except ValueError:
            print('[%s] already exists, using it directly.' % S)
            s_group = hdf5[S]

        try:
            roi_group = s_group.create_group(R)
        except ValueError:
            print('[%s] already exists, using it directly.' % R)
            roi_group = s_group[R]

        try:
            layer_group = roi_group.create_group(L)
        except ValueError:
            print('[%s] already exists, using it directly.' % L)
            layer_group = roi_group[L]

        try:
            sub = layer_group.create_group(norm_tec[N])

        except ValueError:
            print('[%s] already exists, using it directly.' % norm_tec[N])
            sub = layer_group[norm_tec[N]]

        for D in dataTypes:

            if D != 'alpha' and D != 'weight' and D != 'gain':
                try:
                    sub.create_dataset(D, data=data_dict[D])
                except RuntimeError:
                    del sub[D]
                    sub.create_dataset(D, data=data_dict[D])

            else:
                try:
                    group = sub.create_group(D)
                except ValueError:
                    print('Group [%s] already exists, using it directly.' % D)
                    del sub[D]
                    group = sub.create_group(D)

                for i in range(200):
                    mark = str(i)
                    try:
                        group.create_dataset(mark, data=data_dict[D][mark])
                    except RuntimeError:
                        del group[D][mark]
                        group.create_dataset(mark, data=data_dict[D][mark])

        print('Data Collected. (。・∀・)ノ\n')

        hdf5.close()

    def save_merged_result(self, data_dict: dict, sbj_num: str):
        print('Save to mergedResult.hdf5 !!!!!!')
        output = h5py.File(self.MResult_path, 'a')

        try:
            loc = output.create_group(sbj_num)
        except ValueError:
            print("Directory ['%s'] already exists, using it directly." % sbj_num)
            loc = output[sbj_num]

        r_keys = list(data_dict.keys())
        for R in r_keys:
            l_keys = list(data_dict[R].keys())

            for L in l_keys:
                n_keys = list(data_dict[R][L].keys())

                for N in n_keys:
                    d_keys = list(data_dict[R][L][N].keys())

                    for D in d_keys:
                        try:
                            r_file = loc.create_group(R)
                        except ValueError:
                            r_file = loc[R]

                        try:
                            l_file = r_file.create_group(L)
                        except ValueError:
                            l_file = r_file[L]

                        try:
                            n_file = l_file.create_group(N)
                        except ValueError:
                            n_file = l_file[N]

                        try:
                            n_file.create_dataset(D, data=data_dict[R][L][N][D])
                        except RuntimeError:
                            del n_file[D]
                            n_file.create_dataset(D, data=data_dict[R][L][N][D])

        output.close()

    def path_setting(self, mode: str, path: str):
        if mode == 'data':
            self.data_path = path
        elif mode == 'result':
            self.result_path = path
        elif mode == 'std':
            self.XY_T_STD_path = path
        elif mode == 'merged':
            self.MResult_path = path
        elif mode == 'hdf5':
            self.hdf5_path = path
        elif mode == 'cat':
            self.cat_data_path = path
