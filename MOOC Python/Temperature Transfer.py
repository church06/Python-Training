# 温度转换
temp = input('Input temperature with unit: ')
def tc (n):
    if tc [-1] in ['F', 'f']:
        print((eval(temp[0:-1]) - 32) / 1.8, 'C')
    elif tc [-1] in ['C', 'c']:
        print((eval(temp [0:-1]) + 32) * 1.8, 'F')
    else:
        print('Error')
print(tc(temp))