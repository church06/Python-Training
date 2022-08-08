import time
long = 10
print('{0:-^30}'.format('Start'))
for i in range(long + 1):
    a = i * '*'
    b = (long - i) * '.'
    c = (i / long) * 100
    print('\r{:^3.0f}%[{}->{}]'.format(c, a, b), end = '')
    time.sleep(0.1)
print('\n' + '{0:-^30}'.format('End'))
print('\n')

scale = 50
print('Start'.center(scale//2, '-'))
s = time.perf_counter()
for i in range(scale + 1):
    a = i * '*'
    b = (scale - i) * '.'
    c = (i / scale) * 100
    dur = time.perf_counter() - s
    print('\r{:^3.1f}%[{}->{}]{:.2f}s'.format(c, a, b, dur), end = '')
    time.sleep(0.1)
print('\n' + 'End'.center(scale//2, '-'))