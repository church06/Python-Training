import turtle as tt
import time
def DrawGap():
    tt.penup()
    tt.fd(5)
def Drawline(draw):
    DrawGap()
    tt.pendown() if draw else tt.penup()
    tt.fd(40)
    DrawGap()
    tt.right(90)
def DrawDigit(digit):
    Drawline(True) if digit in [2, 3, 4, 5, 6, 8, 9] else Drawline(False)
    Drawline(True) if digit in [0, 1, 3, 4, 5, 6, 7, 8, 9] else Drawline(False)
    Drawline(True) if digit in [0, 2, 3, 5, 6, 8, 9] else Drawline(False)
    Drawline(True) if digit in [0, 2, 6, 8] else Drawline(False)
    tt.left(90)
    Drawline(True) if digit in [0, 4, 5, 6, 8, 9] else Drawline(False)
    Drawline(True) if digit in [0, 2, 3, 5, 6, 7, 8, 9] else Drawline(False)
    Drawline(True) if digit in [0, 1, 2, 3, 4, 7, 8, 9] else Drawline(False)
    tt.left(180)
    tt.penup()
    tt.fd(20)
def drawDate(date):
    for i in date:
        if i == '-':
            tt.write('年', font=('Arial', 50, 'normal'))
            tt.pencolor('green')
            tt.fd(80)
        elif i == '=':
            tt.write('月', font=('Arial', 50, 'normal'))
            tt.pencolor('blue')
            tt.fd(80)
        elif i == '+':
            tt.write('日', font=('Arial', 50, 'normal'))
            tt.pencolor('red')
        else:
            DrawDigit(eval(i))
def main():
    tt.setup(1000, 350, 200, 200)
    tt.penup()
    tt.fd(-405)
    tt.pensize(5)
    drawDate(time.strftime('%Y-%m=%d+', time.gmtime()))
    tt.hideturtle()
    tt.done()
main()