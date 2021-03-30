count = 0
def hanoi(n, frt, lst, mid):
    global count
    if n == 1:
        print('{:<} | {:>}:{:>}->{:>}'.format(count, n, frt, lst))
        count += 1
    else:
        hanoi(n-1, frt, mid, lst)
        print('{:<} | {:>}:{:>}->{:>}'.format(count, n, frt, lst))
        count += 1
        hanoi(n-1, mid, lst, frt)
a = eval(input('number of disks: '))
def main():
    hanoi(a, 'A', 'B', 'C')
main()
