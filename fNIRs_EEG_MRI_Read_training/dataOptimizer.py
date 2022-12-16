import pandas
import numpy


class Optimize:

    def __init__(self):
        self.path = ''
        self.sub = 1
        self.formate = 'snirf'

    def select_data(self, data_path: str, target_subject: int, data_formate: str):
        self.path = data_path
        self.sub = target_subject
        self.formate = data_formate

    def __read_data(self):
        data_sub = {1: 'sub-01', 2: 'sub-02',
                    3: 'sub-03', 4: 'sub-04',
                    5: 'sub-05'}

        data_type = {'snirf': '//nirs//sub-{0}_task-tapping_nirs.snirf'.format('0' + str(self.sub)),
                     'tsv': '//sub-{0}_scans.tsv'.format('0' + str(self.sub))}

        return self.path + data_sub[self.sub] + data_type[self.formate]

    def optimize(self):
        print('Optimize data on: {}'.format(self.path))

        formulated = pandas.DataFrame()

