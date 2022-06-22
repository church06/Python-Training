class GetData:

    def __init__(self):
        self.root_path = 'G:/Entrance/Coding/datasets/MRI/GOD_rawData'

        self.target_sub = {1: 'sub-01',
                           2: 'sub-02',
                           3: 'sub-03'}

        self.target_dataType = {'img': 'ses-imageryTest',
                                'pct-test': 'ses-perceptionTest',
                                'pct-train': 'ses-perceptionTraining'}

        self.target_dataNum = {1: '01',
                               2: '02',
                               3: '03',
                               4: '04'}

        self.full_testName = {'img': 'task-imagery',
                              'pct_test': 'task-perception',
                              'pct-train': 'task-perception'}

        self.data_path = ''

    def targetDataPath(self, sub: int, dataType: str, dataNum: int, run: int):
        aim_sub = self.target_sub[sub]
        aim_dataType = self.target_dataType[dataType] + self.target_dataNum[dataNum]
        final_path = self.root_path + '/' + aim_sub + '/' + aim_dataType + '/func/'

        self.data_path = final_path

        dataName = aim_sub + '_' + aim_dataType + '_' + self.full_testName[
            dataType] + '_run-{0}_events.tsv'.format('%02d' % run)

        return final_path + dataName

    def setRootPath(self, path: str):
        self.root_path = path

    def getRootPath(self):
        print(self.root_path)
