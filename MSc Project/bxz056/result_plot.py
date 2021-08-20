import h5py


def read_data(data_dir: str):
    print('Getting File...')
    file = h5py.File(data_dir, 'r')
    print(file.keys())



