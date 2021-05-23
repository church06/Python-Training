import pandas
import sklearn.datasets

file = sklearn.datasets.load_boston()

data = file.data
target = file.target

x = pandas.DataFrame(data, columns=file['feature_names'])
y = pandas.DataFrame(target, columns=['MEDV'])


