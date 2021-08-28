from itertools import product

import bdpy.stats
import h5py
import numpy
from bdpy import get_refdata
import Tools


def create_cat(sbj_num: str):
    print('Creating category data')
    path = 'HDF5s\\'
    data_path = 'data\\'

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
    subject = bdpy.BData(data_path + s_list[sbj_num])
    img_feature = bdpy.BData(data_path + 'ImageFeatures.h5')
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


def read_cate(sbj_num: str):
    path = 'HDF5s\\cat_data.hdf5'
    file = h5py.File(path, 'r')
    sbj = file[sbj_num]

    output = {}

    keys = list(sbj.keys())

    for key in keys:
        output[key] = numpy.array(sbj[key])

    return output


def save_data(data_dict: dict, sbj_num: str):
    print('Save to results.hdf5 !!!!!!')
    output = h5py.File('HDF5s\\mergedResult.hdf5', 'a')

    r_keys = list(data_dict.keys())
    l_keys = list(data_dict[r_keys[0]].keys())
    n_keys = list(data_dict[r_keys[0]][l_keys[0]].keys())
    d_keys = list(data_dict[r_keys[0]][l_keys[0]][n_keys[0]].keys())

    try:
        loc = output.create_group(sbj_num)
    except ValueError:
        print("Directory ['%s'] already exists, using it directly." % sbj_num)
        loc = output[sbj_num]

    for R, L, N, D in product(r_keys, l_keys, n_keys, d_keys):

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


def error_detected(name: str):
    content = 'Group |[{ %s }]| already exists, y/n continue? ' % name

    user_input = input(content).lower().strip()

    while True:
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False

        else:
            user_input = input('Unknown input. ' + content).lower().strip()


def merge_data(file: dict):
    output = {}

    p_pt_av = file['p_pt_av']
    t_pt_av = file['t_pt_av']
    p_im_av = file['p_im_av']
    t_im_av = file['t_im_av']

    cor_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_pt_av, p_pt_av)]
    cor_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_im_av, p_im_av)]
    cor_cat_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_pt_av, p_pt_av)]
    cor_cat_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_im_av, p_im_av)]

    output['cor_pt_av'] = cor_pt_av
    output['cor_im_av'] = cor_im_av
    output['cor_cat_pt_av'] = cor_cat_pt_av
    output['cor_cat_im_av'] = cor_cat_im_av

    return output


# =====================================================================================
create_cat('s1')

data = Tools.read_result_data('s1')
category = Tools.read_cate('s1')

roi_s = list(data.keys())
layer_s = list(data[roi_s[0]].keys())
norm_s = data[roi_s[0]][layer_s[0]]

mergedResult = {}

cat_pt_av = numpy.array(category['cat_pt_av'])
cat_im_av = numpy.array(category['cat_im_av'])

r_dict = {}
for r in roi_s:
    l_dict = {}

    for f in layer_s:
        norm_dict = {}

        for n in norm_s:
            target = data[r][f][n]

            m_dict = merge_data(target)
            norm_dict[n] = m_dict

        l_dict[f] = norm_dict
    mergedResult[r] = l_dict

save_data(mergedResult, 's1')
