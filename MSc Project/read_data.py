import os.path

import bdpy.bdata
import h5py
import numpy
from bdpy.bdata import concat_dataset

subjects = {'s1': os.path.abspath('data/Subject1.h5'),
            's2': os.path.abspath('data/Subject2.h5'),
            's3': os.path.abspath('data/Subject3.h5'),
            's4': os.path.abspath('data/Subject4.h5'),
            's5': os.path.abspath('data/Subject5.h5'),
            'imageFeature': os.path.abspath('data/ImageFeatures.h5')}

regine_of_interest = {'VC': 'ROI_VC = 1',
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

Image_Feature = 'data/ImageFeatures.h5'

print('=======================================')
print('Data loading...')

data = {}
keys = {}
image_feature = {}

for person in subjects:

    file = h5py.File(subjects[person], 'r')

    if len(subjects[person]) == 1 and person != 's2' and person != 'imageFeature':

        print(person, '---------------------')
        print('data: ', file.keys())

        data[person] = bdpy.BData(subjects[person][0])

    else:
        # Subject2 & image feature
        if person == 'imageFeature':
            image_feature = file.keys()


print(data)
