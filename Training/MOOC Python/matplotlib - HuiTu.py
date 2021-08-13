import matplotlib.pyplot as mp

mp.rcParams['font.family'] = 'SimHei'
fig = mp.figure(facecolor='lightgray')
mp.subplot(2, 2, 1)
mp.title('subtitle1')
mp.subplot(2, 2, 2)
mp.title('subtitle2', loc='left', color='b')
mp.subplot(2, 2, 3)
myfontdict = {'fontsize': 12, 'color': 'g', 'rotation': 30}
mp.title('subtitle3', fontdict=myfontdict)
mp.subplot(2, 2, 4)
mp.title('subtitle4', color='white', backgroundcolor='black')

mp.suptitle('Suptitle', fontsize=20, color='red', backgroundcolor='yellow')

mp.tight_layout(rect=[0, 0, 1, 0.9])

mp.show()
