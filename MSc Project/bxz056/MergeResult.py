import bdpy.stats
import h5py
import numpy
from bdpy import get_refdata


def read_data(path: str):
    print('Getting File...')

    file = h5py.File(path, 'r')

    output = {}
    layer_list = list(file.keys())

    for layer in layer_list:
        iter_list = list(file[layer].keys())

        # collect iterations
        iter_dict = {}
        for iteration in iter_list:
            norm_list = list(file[layer][iteration].keys())

            # collect normalizations
            norm_dict = {}
            for norm in norm_list:
                target = file[layer][iteration][norm]
                type_list = list(target.keys())

                data_dict = {}
                for data_type in type_list:
                    data_dict[data_type] = numpy.array(target[data_type])

                norm_dict[norm] = data_dict
            iter_dict[iteration] = norm_dict
        output[layer] = iter_dict

    file.close()

    return output


def create_cat():
    print('Creating category data')
    path = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\HDF5s\\'
    data_path = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\data\\'

    out_file = h5py.File(path + 'cat_data.hdf5', 'a')
    groups = ['cat_pt_av', 'cat_im_av', 'cat_pt_label', 'cat_im_label']

    # Getting data ---------------------------------------
    subject = bdpy.BData(data_path + 'Subject1.h5')
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

    # Get averaged predicted feature
    test_label_pt = labels[i_test_seen, :].flatten()
    test_label_im = labels[i_test_im, :].flatten()

    # Get category averaged features
    cat_labels_pt = numpy.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
    cat_labels_im = numpy.vstack([int(n) for n in test_label_im])  # Category labels (perception test)

    cat_pt_label = numpy.unique(cat_labels_pt)
    cat_im_label = numpy.unique(cat_labels_im)

    cat_labels = img_feature.select('CatID')  # Category labels in image features
    ind_cat_av = (img_feature.select('FeatureType') == 3).flatten()

    cat_pt_av = get_refdata(layer_feature[ind_cat_av, :], cat_labels[ind_cat_av, :], cat_pt_label)
    cat_im_av = get_refdata(layer_feature[ind_cat_av, :], cat_labels[ind_cat_av, :], cat_im_label)
    # ----------------------------------------------------------------------------------------------------------

    temp_dict = {groups[0]: cat_pt_av, groups[1]: cat_im_av,
                 groups[2]: cat_pt_label, groups[3]: cat_im_label}

    for name in groups:
        try:
            print("Create ['%s']." % name)
            out_file.create_dataset(name, data=temp_dict[name])
        except RuntimeError:
            print("['%s'] already exists. Remove file and get new data" % name)
            del out_file[name]
            out_file.create_dataset(name, data=temp_dict[name])

    out_file.close()


def read_cate(path: str):
    file = h5py.File(path, 'r')

    output = {}

    keys = list(file.keys())

    for key in keys:
        output[key] = numpy.array(file[key])

    return output


def save_data(path: str, feature: str, data_dict: dict):
    print('Save to results.hdf5 !!!!!!')

    output = h5py.File(path + 'mergedResult.hdf5', 'a')

    try:
        loc = output.create_group(feature)
    except ValueError:
        print("['%s'] already exists, using it directly.")
        loc = output[layer]

    iter_keys = list(data_dict.keys())
    norm_keys = list(data_dict[iter_keys[0]].keys())
    data_keys = list(data_dict[iter_keys[0]][norm_keys[0]].keys())

    for i_key in iter_keys:
        print("Create ['%s']" % i_key)

        try:
            i_loc = loc.create_group(i_key)

        except ValueError:
            print("Create ['%s']" % i_key)
            i_loc = loc[i_key]

        for n_key in norm_keys:
            print("Create ['%s']" % n_key)

            try:
                n_loc = i_loc.create_group(n_key)

            except ValueError:
                print("Create ['%s']" % n_key)
                n_loc = i_loc[n_key]

            t_dict = data_dict[i_key][n_key]
            for d_key in data_keys:
                n_loc.create_dataset(d_key, data=t_dict[d_key])

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


# =====================================================================================

hdf5_dir = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\HDF5s\\'
result_file = hdf5_dir + 'results.hdf5'
cat_file = hdf5_dir + 'cat_data.hdf5'

create_cat()

data = read_data(path=result_file)
category = read_cate(path=cat_file)

iter_s = ['iter_50', 'iter_100', 'iter_150', 'iter_200']
norm_s = ['none', 'z-score', 'min-max', 'decimal']
layer = 'cnn1'

location = data[layer]
mergedResult = {}

cat_pt_av = numpy.array(category['cat_pt_av'])
cat_im_av = numpy.array(category['cat_im_av'])

for iteration in iter_s:

    norm_dict = {}
    for norm in norm_s:
        value_dict = {}

        target = location[iteration][norm]

        p_pt_av = target['p_pt_av']
        t_pt_av = target['t_pt_av']
        p_im_av = target['p_im_av']
        t_im_av = target['t_im_av']

        cor_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_pt_av, p_pt_av)]
        cor_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_im_av, p_im_av)]
        cor_cat_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_pt_av, p_pt_av)]
        cor_cat_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_im_av, p_im_av)]

        value_dict['cor_pt_av'] = cor_pt_av
        value_dict['cor_im_av'] = cor_im_av
        value_dict['cor_cat_pt_av'] = cor_cat_pt_av
        value_dict['cor_cat_im_av'] = cor_cat_im_av

        norm_dict[norm] = value_dict

    mergedResult[iteration] = norm_dict

save_data(hdf5_dir, layer, mergedResult)
