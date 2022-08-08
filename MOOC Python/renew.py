import time as tm
for i in range(101):
    print('\r{:3}%'.format(i), end = '')
    tm.sleep(0.5)