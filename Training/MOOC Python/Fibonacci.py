def F(n):
    if n == 1 or n == 2:
        return 1
    else:
        return F(n - 1) + F(n - 2)

n = eval(input('enter a number: '))
print(F(n))
