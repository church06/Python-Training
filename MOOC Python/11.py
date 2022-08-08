a = eval(input('a = '))
b = 'Hello World'
c = 0
d = ''
if a == 0:
    print(b)
elif a > 0:
    for i in b:
        d += i
        if len(d) == 2:
            print(d)
            d = ''
    print('d')
else:
    for i in b[::1]:
        print(i)
# -------------------------------------
rst = eval(input())
print('{:.2f}'.format(rst))
h = 'dfgdfg'
print(h.strip('d'))
# --------------------------------------
v = eval(input())
print('{:+>30.3f}'.format(pow(v, 0.5)))
# --------------------------------------
s = input()
m = s.split('-')
print('{}{}{}'.format(m[0], '+', m[-1]))