class fNIRs_DataReader:
    def __init__(self):
        self.data_path = "G://Entrance//Coding//datasets//fNIRs//rob-luke-BIDS-NIRS-Tapping//"

        self.subs = {'01': 'sub-01//nirs//sub-01_task-tapping_nirs.snirf',
                     '02': 'sub-02//nirs//sub-02_task-tapping_nirs.snirf',
                     '03': 'sub-03//nirs//sub-03_task-tapping_nirs.snirf',
                     '04': 'sub-04//nirs//sub-04_task-tapping_nirs.snirf',
                     '05': 'sub-05//nirs//sub-05_task-tapping_nirs.snirf'}

    def set_dataPath(self, path: str):
        self.data_path = path
        print('Data path set to: ' + path)

    def get_dataPath(self):
        print('Data path: ' + self.data_path)
        return self.data_path

    def set_subs(self, sub: dict):
        self.subs = sub
        print('Subjects set to: ')

        for key in sub:
            print('{0} | {1}'.format(key, sub[key]))

    def get_subs(self):
        print('Getting Subjects...')
        print('============================================')

        for key in self.subs:
            print('{0} | {1}'.format(key, self.subs[key]))

        print('============================================\n')
        return self.subs

    def get_data(self, sub: str):
        key = list(self.subs.keys())

        if sub in key:
            path_full = self.data_path + self.subs[sub]
            print('Getting Data: Subject-' + sub)
            print('Path: ' + path_full + '\n')
            return path_full
        else:
            print('Could not find Data.\n')
