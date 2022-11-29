import pandas


def read_data(disk: str, sub: int, formate: str):
    data_path = disk + '//Entrance//Coding//datasets//fNIRs//rob-luke-BIDS-NIRS-Tapping//'

    data_sub = {1: 'sub-01', 2: 'sub-02',
                3: 'sub-03', 4: 'sub-04',
                5: 'sub-05'}

    data_type = {'snirf': '//nirs//sub-{0}_task-tapping_nirs.snirf'.format('0' + str(sub)),
                 'tsv': '//sub-{0}_scans.tsv'.format('0' + str(sub))}

    return data_path + data_sub[sub] + data_type[formate]


