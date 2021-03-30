a = 0
while a < 1:
    week = '星期一星期二星期三星期四星期五星期六星期日'
    id = eval(input('input number: '))
    if 1<= id <= 7:
        pos = (id - 1) * 3
        print(week[pos : pos + 3])
    else:
        print('Please provide reasonable value')