import pandas
import getDataClass

getData = getDataClass.GetData()
file = getData.targetDataPath(1, 'img', 1, 1)
print(file)
print('=================================================================================')

data = pandas.read_csv(file, sep='\t', header=0)
print(data)
