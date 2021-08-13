import numpy as np

# --------------------------------------------------------

print('数组')

# --------------------------------------------------------

print('创建数组')
# array([列表]/(元组))
print(np.array([0, 1, 2, 3]))
print(np.array((1, 2, 3)))
a = np.array([[0, 1], [2, 3], [4, 5]])
print(type(a))

print('输出维数')
print(a.ndim)

print('输出形状')
print(a.shape)

print('输出元素个数')
print(a.size)

print('输出特定维度')
print(a[0])
print(a[0, 1])

# --------------------------------------------------------

print('创建由数字序列构成的数组')
# np.arange(起始数字, 结束数字, 步长, dtype=数据类型), 前闭后开
print(np.arange(0, 2, 0.3))

print('创建全部元素为1数组')
# np.ones(shape, dtype=数据类型)
print(np.ones((3, 2), dtype=np.int16))
print(np.ones((3, 2)))

print('创建全部元素为0数组')
# np.zeros(shape, dtype=数据类型)
print(np.zeros((2, 3)))

print('创建单位矩阵')
# np.eye(shape, dtype=数据类型)
print(np.zeros(2))
print(np.zeros((2, 3)))

print('创建等差数组')
# np.linspace(start, stop, num=50)
print(np.linspace(1, 10, 10))

print('创建等比数列')
# np.logspace(start, stop, num=50, base=10)
print(np.logspace(1, 10, 10, base=2))

# --------------------------------------------------------

print('将列表或元组转化成数组对象')
# asarray()
arr1 = np.ones((3, 3))
arr2 = np.array(arr1)
arr3 = np.asarray(arr1)

arr1[0][0] = 3
print('arr1: \n', arr1)
print('arr2: \n', arr2)
print('arr3: \n', arr3)

print('将数组切片')
b = np.array([[[101, 203, 440], [412, 34, 534], [413, 34543, 6474], [57867, 324, 2423]],
              [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
print(b[0, 0])
print(b[:, :, 0])
print(b[0:2, 1:3])

print('不改变当前数组, 照shape创建新数组')
# np.reshape(shape)
c = np.arange(12)
c.reshape(3, 4)
c.resize(3, 4)
d = np.arange(45).reshape(3, 5, 3)
print(d)
c.reshape(-1, 1)
d.reshape(-1)

# --------------------------------------------------------

print('数组相加')
# 同元素个数可相加，否则报错
# 一维数组可与多维数组相加
# 减乘除规则与加相同
e = np.array([0, 1, 2])
f = np.arange(3, 6)
print(e + f)
g = np.array([1.1, 2.3, 3.1])
h = np.linspace(3, 4, 3)
print(g + h)

# --------------------------------------------------------

print('数组元素切片')
print('一维数组')
i = np.array([1, 2, 3])
print(i[0])
print(i[0:2])
print(i[0:])
print('\n')
print('二维数组')
j = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(j[0])
print(j[0: 2], j[:2])
print(j[0: 2, 0: 2])
print(j[0: 2, 1: 3])
print('三维数组')
k = np.array([[[123, 313, 345, 6456], [16551, 4561, 87856, 35135], [158111, 879885, 35435, 4568]],
              [[46859, 4658, 2451, 5280], [52846, 2541, 5241, 556274], [57860, 75684, 72504, 452]]])
print(k[:, :, 0])
print(k[:, :, 1])
print(k[:, 0, :])
print(k[0, :, :])

print('不改变当前数组, 按照shape创建新数组')
# np.reshape(shape)
print('直接修改数组至size形状')
# np.resize(size)
# 需确保形状与元素总数相等
l = np.arange(12)
print(l.reshape(3, 4))
print(l.resize(3, 4))

print('reshape()取负数则自动取值')
print(l.reshape(-1, 1))
print(l.reshape(-1))

print('创建并改变数组形状')
print(np.arange(15).reshape(5, 3))

# 数组相加形状应一致, 否则报错
print('减乘除与加法规则一致')
m = np.arange(3)
n = np.array(([7, 5, 8]))
print(m + n)
o = np.array([[1, 2, 3], [6, 7, 5], [4, 2, 6]])
print(n + o)

print('矩阵相乘')
# np.dot(matrix1, matrix2)
# np.matmul(matrix1, matrix2)
p = np.array([[2, 3], [4, 5]])
q = np.array([[15, 4], [514, 4578]])
print(np.dot(p, q))
print(np.matmul(p, q))

print('转置与求逆')
# 转置: np.transpose()
# 求逆: np.linalg.inv()
print(np.transpose(p))
print(np.linalg.inv(p))

# --------------------------------------------------------

print('矩阵与随机数')
# np.matrix(string/list/tuple/array)
# np.mat(string/list/tuple/array)
r = np.mat('1 2 3 ; 4 5 6')
print(r)
s = np.array([[1, 2, 3], [4, 5, 6]])
t = np.mat(a)

print('矩阵相乘')
u = np.mat([[0, 1], [45, 7]])
v = np.mat(np.array([[234, 67], [5614, 984]]))
print(v * u)

print('矩阵转置与求逆')
# .T - 转置
# .I - 求逆
print(u.T)
print(u.I)

print('随机数')
# np.random.rand(d0, ..., dn)          - 元素在[0, 1)间均匀分布的数组                         返回值: 浮点数
# np.random.uniform(low, hige, size)   - 元素在[low, hige)区间均匀分布的数组                  返回值: 浮点数
# np.random.randint(low, hige, size)   - 元素在[low, hige)区间均匀分布的数组                  返回值: 整数
# np.random.randn(d0, ..., dn)         - 产生标准正态分布的数组                               返回值: 浮点数
# np.random.normal(loc, scale, size)   - 产生正态分布数组 (log = 均值, scale = 标准差)         返回值: 浮点数

print(np.random.rand(2, 3))
print('参数为空则返回数字')
print(np.random.rand())
print('同seed, 同随机数')
np.random.seed(54)
print(np.random.rand(4, 5))
np.random.seed(54)
print(np.random.rand(4, 5))
print(np.random.rand(4, 5))

print('均匀分布')
print(np.random.uniform(1, 10, (5, 5)))
print('整数均匀分布')
print(np.random.randint(1, 10, (5, 5)))
print('标准正态分布')  # 标准正态分布: 标准差为1, 均值为0
print(np.random.randn(5, 5))
print('正态分布')
print(np.random.normal(15, 10, (5, 5)))

# --------------------------------------------------------

print('打乱顺序')
# np.random.shuffle(sequence)
w = np.arange(15)
np.random.shuffle(w)
print(w)
np.random.shuffle(w)
print(w)

print('对于数组, shuffle只打乱样本集排序, 不改变数组内部顺序')
x = np.arange(15).reshape(3, 5)
print(x)
np.random.shuffle(x)
print(x)

# --------------------------------------------------------

print('数据可视化')
import matplotlib.pyplot as plt

# --------------------------------------------------------

print('创建对象')
# figure(num, figsize, dpi, facecolor, edgecolor, frameon)

# num       - 图形编号或名称, 编号取值为数字, 名称为字符串
# figsize   - 绘图对象宽高, 单位: 英寸
# dpi       - 绘图对象分辨率, 缺省则为80
# facecolor - 背景颜色
# edgecolor - 边框颜色

plt.figure(figsize=(3, 2), facecolor='green')
plt.plot()
plt.show()

print('划分子图')
# subplot(line, list, mark)
y = plt.figure()
plt.subplot(2, 2, 1)
plt.subplot(2, 2, 2)
plt.subplot(2, 2, 3)
plt.subplot(2, 2, 4)
plt.show()

z = plt.figure()
plt.subplot(321)
plt.subplot(322)
plt.subplot(323)
plt.subplot(324)
plt.subplot(325)
plt.subplot(326)
plt.show()

# 设置中文字体
# plt.rcParams['font.sans-serif']='SimHei'
# 恢复默认配置
# plt.rcdefaults()
# 添加标题
# 全局: suptitle()
# 子图: title()

#           参数                     说明                               默认值

# loc      - (title() only)     - 标题位置                            left/right
# rotation - (title()only)      - 标题旋转角度
# x                             - 标题x坐标                              0.5
# y                             - 标题y坐标                              0.98
# color                         - 标题颜色                               black
# backgroundcolor               - 标题背景颜色                             12
# fontsize                      - 标题字体大小     xx-small/x-small/small/medium/large/x-large/xx-large
# fontweight                    - 字体粗细            light/normal/medium/semibold/bold/heavy/black
# fontstyle                     - 字体类型                       normal/italic/oblique
# horizontalalignment           - 标题水平对齐方式                  left/center/right
# verticaltalalignment          - 标题垂直对齐方式              top/center/bottom/baseline
# fontdict                      - 设置参数字典

plt.rcParams['font.family'] = 'SimHei'

A = plt.figure(facecolor='lightgray')
plt.subplot(221)
plt.title('111111')
plt.subplot(222)
plt.title('222222', loc='left', color='b')
plt.subplot(223)
plt.title('333333', fontsize=12, color='g', rotation=30)
plt.subplot(224)
plt.title('444444', color='white', backgroundcolor='black')

plt.suptitle('11111', fontsize=28, color='red', backgroundcolor='yellow')

# 图像自动调整
# tight_layout(rect=[left, bottom, right, top])

plt.tight_layout()

plt.show()

# --------------------------------------------------------

# 散点图
# scatter(x, y, scale, color, marker, label)
# x       - 数据x坐标(不可省略)
# y       - 数据y坐标(不可省略)
# scale   - 数据点大小
# color   - 数据点颜色
# marker  - 数据点样式(默认'o')
# label   - 图例文字

# 字体
plt.rcParams['font.sans-serif'] = 'SimHei'

# 显示负号
plt.rcParams['axes.unicode_minus'] = False

# 正态分布
B = 1024
C = np.random.normal(0, 1, B)
D = np.random.normal(0, 1, B)

C1 = np.random.normal(-4, 4, (1, B))
D1 = np.random.normal(-4, 4, (1, B))

# 绘制
plt.scatter(C, D, color='blue', marker='*')
plt.scatter(C1, D1, color='red', marker='o')

# 标题
plt.title('标准正态分布', fontsize=20)

# 指定位置添加文字
# plt.text(x, y, s, fontsize, color)
# s - 显示的文字
# x, y, s 不可省略
plt.text(2.5, 2.5, '均  值: 0\n标准差: 1')

# 坐标轴设置
# plt.rcParams['axes.unicode_minus']
# xlabel(x, y, s, fontsize, color) - x轴标签
# ylabel(x, y, s, fontsize, color) - y轴标签
# xlim(xmin, xmax)                 - x坐标轴范围
# ylim(ymin, ymax)                 - y坐标轴范围
# tick_params(labelsize)           - 刻度文字字号
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('横坐标x', fontsize=14)
plt.ylabel('横坐标y', fontsize=14)

plt.show()

# --------------------------------------------------------

# 折线图&柱形图
# plt.plot(x, y, color, marker, label, linewidth, markersize)
# linewidth - 折线宽度

plt.rcParams['font.sans-serif'] = 'SimHei'

E = 24
A1 = np.random.randint(27, 37, E)
B1 = np.random.randint(40, 60, E)

plt.plot(A1, label='温度')
plt.plot(B1, label='湿度')

plt.xlim(0, 23)
plt.ylim(20, 70)
plt.xlabel('小时', fontsize=14)
plt.ylabel('温度/湿度', fontsize=14)

plt.suptitle('温湿度24h记录', fontsize=20)

plt.legend()  # 显示图例
plt.show()

# --------------------------------------------------------

# 柱形图
# plt.bar(left, high, width, facecolor, edgecolor, label)
# left - 柱形条左边沿位置
# facecolor - 条纹填充颜色
# edgecolor - 条纹边缘颜色

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 条纹高度
E1 = [32, 25, 16, 30, 24, 45, 40, 33, 28, 17, 24, 20]
F1 = [-23, -35, -26, -35, -45, -43, -35, -32, -23, -17, -22, -28]
# 条纹左侧坐标
plt.bar(range(len(E1)), E1, width=0.8, facecolor='green', edgecolor='white', label='统计量1')
plt.bar(range(len(F1)), F1, width=0.8, facecolor='red', edgecolor='white', label='统计量2')

plt.title('柱状图', fontsize=20)
plt.legend()
plt.show()

# --------------------------------------------------------

# 波士顿房价分析

# Keras数据集

# 名称               说明
# bonston_housing   - 波士顿房价数据集
# CIFAR10           - 10种类别图片集
# CIFAR100          - 100种类别图片集
# MNIST             - 手写数字图片集
# Fashion-MNIST     - 10种时尚类别图片集
# IMDB              - 电影点评数据集
# reuters           - 路透社新闻数据集

# 变量名          说明
# CRIM     - 城镇人均犯罪率
# ZN       - 超过2.5W平方英尺住宅所占比例
# INDUS    - 城镇非零售业商业用地比例
# CHAS     - 是否被Charles河穿过(1: True, 0:False)
# NOX      - 一氧化碳浓度
# RM       - 住宅平均房间数
# AGE      - 早于1940年建成的自住房比例
# DIS      - 到波士顿5个中心区域的加权平均距离
# RAD      - 到达高速公路的便利指数
# TAX      - 每1w美元的全值财产税率
# PTRATIO  - 城镇中师生比例
# B        - 城镇中黑人比例, 越靠近0.63越小; B = 1000*(BK - 0.63)², BK为黑人比例
# LSTAT    - 低收入人口比例
# MEDV     - 自住房平均房价($000)

# 数据集加载: tensorflow.keras.datasets.数据集名称
import tensorflow as tf

F = tf.keras.datasets.boston_housing
# 数据加载, test_split=num划分训练集和测试集
(train_x, train_y), (test_x, test_y) = F.load_data(test_split=0.3)

print('Training set:', len(train_x))
print('Testing set:', len(test_x))

plt.figure(figsize=(5, 5))
plt.scatter(train_x[:, 5], train_y)
plt.xlabel('RM')
plt.ylabel('Price($000')
plt.title('5. Price-RM')
plt.show()

# 所有属性与房价间的关系
titles = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B-1000', 'LSTAT', 'MEDV']
plt.figure(figsize=(12, 12))
for i in range(13):
    # 划分子图
    plt.subplot(4, 4, (i + 1))
    plt.scatter(train_x[:, i], train_y, 5)

    plt.xlabel(titles[i])
    plt.ylabel('Price($000)')
    plt.title(str(i + 1) + '.' + titles[i] + ' - Price')

plt.show()

# --------------------------------------------------------

# Iris 数据集
# 下载数据集: tf.keras.utils.get_file(fname, origin, cashe_dir)
# fname      - 下载后文件名
# origin     - 文件URL
# cashe_dir  - 存储位置

tf.keras.utils.get_file(fname='iris_training.csv', origin='http://download.tensorflow.org/data/iris_training.csv',
                        cache_dir='G:\Entrance\Tensor Data')
# 下载地址调用
tensordatalink = 'http://download.tensorflow.org/data/iris_training.csv'.split('/')[-1]

# --------------------------------------------------------

# pillow 图像处理
import PIL.Image as pil

# image.open()    - 打开图像
# image.save()    - 保存图像

image = pil.open('test.jpg')
image.save('test.png')

# image.format    - 图像格式
# image.size      - 图像尺寸
# image.mode      - 色彩模式
# 取值
# 1      - 二值图像
# L      - 灰度图像
# P      - 8位彩色图像
# RGB    - 24位彩色图像
# RGBA   - 32位彩色图像
# CMYK   - CMYK彩色图像
# YCbCr  - YCbCr彩色图像
# I      - 32位整型灰度图像
# F      - 32位浮点灰度图像

# plt.imshow()    - 显示图像
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.show()

# 转化为数组
arr_img = np.array(image)
print('shape:', arr_img.shape, '\n')
print(arr_img)

# 图像缩放
# 图像对象.resize((width, height))
img_small = image.resize((64, 64))
plt.imshow(img_small)
plt.show()

image.save('img_small.jpg')

# 图像旋转 & 镜像
plt.figure(figsize=(20, 20))

plt.subplot(331)
plt.axis('on')
plt.imshow(image)
plt.title('Original', fontsize=80)

# 镜像
plt.subplot(332)
plt.axis('off')
image_flr = image.transpose(pil.FLIP_LEFT_RIGHT)
plt.imshow(image_flr)
plt.title('Left to Right', fontsize=60)

plt.subplot(333)
plt.axis('off')
image_ftb = image.transpose(pil.FLIP_TOP_BOTTOM)
plt.imshow(image_ftb)
plt.title('Top to Bottom', fontsize=60)

# 旋转
plt.subplot(334)
plt.axis('off')
image_r90 = image.transpose(pil.ROTATE_90)
plt.imshow(image_r90)
plt.title('Rotate_90', fontsize=80)

plt.subplot(335)
plt.axis('off')
image_r180 = image.transpose(pil.ROTATE_180)
plt.imshow(image_r180)
plt.title('Rotate_180', fontsize=80)

plt.subplot(336)
plt.axis('off')
image_r270 = image.transpose(pil.ROTATE_270)
plt.imshow(image_r270)
plt.title('Rotate_270', fontsize=80)

# 转置
plt.subplot(337)
plt.axis('off')
image_tsps = image.transpose(pil.TRANSPOSE)
plt.imshow(image_tsps)
plt.title('Transpose', fontsize=80)

plt.subplot(338)
plt.axis('off')
image_tsvs = image.transpose(pil.TRANSVERSE)  # 转置后水平翻转
plt.imshow(image_tsvs)
plt.title('Transverse', fontsize=80)

# 裁剪
plt.subplot(339)
plt.axis('off')
img_cro = image.crop((100, 100, 300, 300))
plt.imshow(img_cro)
plt.title('Crop', fontsize=80)

plt.show()

# --------------------------------------------------------

# 手写数字数据集MNIST
import keras as krs

# krs.utils.get_file(fname, origin, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
# fname
# origin
# extract
krs.utils.get_file('mnist', 'http://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', md5_hash=None,
                   file_hash=None, cache_subdir='G:\Entrance\Tensor Data\Keras datasets', hash_algorithm='auto',
                   extract=False, archive_format='auto', cache_dir=None)
mnist = krs.datasets.mnist
(train_x1, train_y1), (test_x1, test_y1) = mnist.load_data()

for i in range(4):
    num = np.random.randint(1, 60000)
    plt.subplot(2, 2, i + 1)
    plt.axis('off')
    plt.imshow(train_x1[num], cmap='gray')
    plt.title(train_y1[num])
plt.show()

# --------------------------------------------------------

# Tensorflow

# --------------------------------------------------------

print('创建Tensor张量')
# tf.constant(value, dtype, shape)

# value   - number/Python list/Numpy array
# dtype   - 元素数据类型

# tf.int8           - 8位有符号整数
# tf.int16          - 16位有符号整数
# tf.int32          - 32位有符号整数
# tf.int64          - 64位有符号整数
# tf.uint8          - 8位无符号整数
# tf.float32        - 32位浮点
# tf.float64        - 64位浮点
# tf.string         - 字符串
# tf.bool           - 布尔型
# tf.complex64      - 复数, 实部与虚部均为32位浮点型

# shape   - 张量形状
G = tf.constant([[10, 5, 80], [12, 5614, 4678]], dtype=tf.int64)
H = tf.constant([[9, 678, 452], [2, 4, 564]], dtype=tf.float64)
print(G, type)
print(H, type)
# 可使用numpy方法
# G.numpy()
# 创建浮点数数组时, numpy默认浮点数类型为64位
# 数据转换通常由低到高,反之容易引发错误导致错误结果

print('元素数据类型转换')
# tf.cast()
print(tf.cast(G, tf.float64), tf.cast(H, tf.int64))
I = tf.constant([-1, 0, 1, 2, 3])
print(tf.cast(I, tf.bool))  # 非0在转换成布尔值时均视为True

print('创建张量 - 2')
# tf.convert_to_tensor(array/list/number/string/bool)
J = np.arange(12).reshape(3, 4)
K = tf.convert_to_tensor(J)
print(type(J))
print(type(K))

print('是否是张量判断')
# is_tensor()
print(tf.is_tensor(J))
print(tf.is_tensor(K))

# 判断是否属于给定类型
# isinstance(object, type)
print('J is Tensor:', isinstance(J, tf.Tensor))
print('J is array:', isinstance(J, np.ndarray))
print('K is Tensor:', isinstance(K, tf.Tensor))
print('K is array:', isinstance(K, np.ndarray))

print('全0/全1张量')
# tf.zeros(shape, dtype=tf.float32)
# tf.ones(shape, dtype=tf.float32)
L = tf.zeros([4, 3], dtype=tf.float64)
M = tf.ones(shape=(5, 2), dtype=tf.float32)
print(L, '\n' * 2, M)

print('同元素值张量')
print(tf.fill([2, 3], 9))
print(tf.fill([2, 3], 9.0))

print('正态分布张量')
# tf.random.truncated_normal(shape, mean, stddev, dtype)
# mean   - 均值
# stddev - 标准差
### 标准差应大于均值

# tf.random.set_seed() - 设置随机种子
tf.random.set_seed(8)
N = tf.random.truncated_normal(shape=(5, 5), mean=4, stddev=8, dtype=tf.float64)
print(N)

print('均匀分布张量')
# tf.random.uniform(shape, minval, maxval, dtype)
O = tf.random.uniform(shape=(6, 6), minval=-9999, maxval=9999, dtype=tf.int64)
print(O)

print('随机打乱')
# 只沿第一维打乱
print(tf.random.shuffle(O))

# --------------------------------------------------------

# 创建序列
# tf.range(start, limit, delta=1, dtype)
# delta - 步长
print(tf.range(1, 10, delta=2, dtype=tf.int64))
print(tf.range(10, delta=2, dtype=tf.int64))

# tf.rank()  - 获取张量维度
# tf.shape() - 获取张量形状
# tf.size()  - 获取张量元素总数
print('维度:', tf.rank(O))
print('形状:', tf.shape(O))
print('元素总数:', tf.size(O))

print('改变张量形状')
# tf.reshape(tensor, shape)
P = tf.range(24)
P = tf.reshape(P, shape=(2, 3, 4))
print(P)

print('增加维度')
# tf.expand_dims(input, axis)
Q = tf.expand_dims(P, 0)
print(Q)

print('删除维度')
# 只能删除长度为1的维度
# tf.squeeze(input, axis)
print(tf.squeeze(Q, 0))

print('交换维度')
# tf.transpose(a, perm)
R = tf.transpose(P, perm=[1, 2, 0])
print(R)

print('拼接张量')
# tf.concat(tensors, axis)
S = tf.convert_to_tensor(G)
T = tf.convert_to_tensor(tf.cast(H, tf.int64))
U = tf.concat([S, T], 0)
print(U)

print('堆叠张量')
# tf.stack(values, axis)
V = tf.stack((S, T), axis=0)
print(V)

print('分解张量')
# tf.unstack(values, axis)
W = tf.unstack(V, axis=0)
print(W)

print('部分采样')
print('V:', V[1, 1, 2])
print('V:', V[1][1][2])
print('T:', T[0][2])
print('T:', T[0, 2])

print('切片')
# tf.range()
X = tf.random.normal([12, 12], mean=15, dtype=tf.float64)
print(X[::2, ::2])

print('数据提取')
# tf.gather(params, axis, indics)
print(tf.gather(G, axis=1, indices=[0, 1]))

print('多点采样')
# tf.gather_nd()
print('X:', tf.gather_nd(X, [[0, 2], [1, 11], [9, 5], [0, 3]]))

# --------------------------------------------------------

print('张量运算')
# tf.add(x, y)        - 加
# tf.subtract(x, y)  - 减
# tf.multiply(x, y)   - 乘
# tf.divide(x, y)     - 除
# tf.math.mod(x, y)   - 取模
# tf.pow(x, y)        - 对x求y的幂次方
# tf.square(x)        - 对x逐元素计算平方根
# tf.sqrt(x)          - 对x逐元素开根号
# tf.exp(x)           - 计算x的e次方
# tf.math.log(x)      - 计算自然对数, 底数为e

print('加法')
print(tf.add(tf.constant([56, 21, 456], dtype=tf.float64), tf.constant([250, 2154, 5230.4], dtype=tf.float64)))
print('+替代tf.add')
print(tf.constant([56, 21, 456], dtype=tf.float64) + tf.constant([250, 2154, 5230.4], dtype=tf.float64))

print('减法')
print(tf.subtract(tf.constant([56, 21, 456], dtype=tf.float64), tf.constant([250, 2154, 5230.4], dtype=tf.float64)))
print('-替代tf.substract')
print(tf.constant([56, 21, 456], dtype=tf.float64) - tf.constant([250, 2154, 5230.4], dtype=tf.float64))

print('乘法')
print(tf.multiply(tf.constant([56, 21, 456], dtype=tf.float64), tf.constant([250, 2154, 5230.4], dtype=tf.float64)))
print('*替代tf.multiply')
print(tf.constant([56, 21, 456], dtype=tf.float64) * tf.constant([250, 2154, 5230.4], dtype=tf.float64))

print('除法')
print(tf.divide(tf.constant([56, 21, 456], dtype=tf.float64), tf.constant([250, 2154, 5230.4], dtype=tf.float64)))
print('/替代tf.divide')
print(tf.constant([56, 21, 456], dtype=tf.float64) / tf.constant([250, 2154, 5230.4], dtype=tf.float64))

print('乘方')
print(tf.pow(tf.constant([56, 21, 456], dtype=tf.int64), tf.constant([2, 2, 5], dtype=tf.int64)))

print('平方')
print(tf.square(tf.constant([56, 21, 456], dtype=tf.int64)))

print('平方根')
print(tf.sqrt(tf.constant([56, 21, 456], dtype=tf.float64)))  # 平方根需用浮点数

print('自然指数')
print(tf.exp(5.0))  # 需用浮点数

print('自然对数')
print(tf.math.log(tf.exp(5.0)))  # 需用浮点数

print('对数运算')
# 利用换底公式
lgrthm_x = tf.constant(256.0)
lgrthm_y = tf.constant(4.0)
print(tf.math.log(lgrthm_x) / tf.math.log(lgrthm_y))

print('对数张量运算')
lgrthm_x1 = tf.constant([1, 6, 9, 16], dtype=tf.float64)
lgrthm_y1 = tf.constant([2, 5, 7, 4], dtype=tf.float64)
print(tf.math.log(lgrthm_x1) / tf.math.log(lgrthm_y1))

# tf.sign(x)         - 返回x符号
# tf.abs(x)          - 对x逐元素求绝对值
# tf.negative(x)     - 对x逐元素求相反数， y=-x
# tf.reciprocal(x)   - 取x的倒数
# tf.logical_not(x)  - 对x逐元素的逻辑非
# tf.ceil(x)         - 向上取整
# tf.floor(x)        - 向下取整
# tf.rint(x)         - 取最接近的整数
# tf.round(x)        - 对x逐元素求舍入最接近的整数
# tf.maximum(x)      - 返回两tensor中的最大值
# tf.minimum(x)      - 返回两tensor中的最小值

# --------------------------------------------------------

print('数据统计')
# tf.reduce_sum(tensor, aixs)   - 求和
# tf.reduce_mean(tensor, axis)  - 求平均值
# tf.reduce_max(tensor, aixs)   - 求最大值
# tf.reduce_min(tensor, axis)   - 求最小值
print('求和')
print('Axis: 0', tf.reduce_sum(tf.constant([[2, 5, 7], [3, 6, 5]]), axis=0))
print('Axis: 1', tf.reduce_sum(tf.constant([[2, 5, 7], [3, 6, 5]]), axis=1))

print('求平均值')
print('Axis: 0', tf.reduce_mean(tf.constant([[2, 5, 7], [3, 6, 5]]), axis=0))
print('Axis: 1', tf.reduce_mean(tf.constant([[2, 5, 7], [3, 6, 5]]), axis=1))
print('Dtype=int (axis: 0): ', tf.reduce_mean(tf.constant([[2, 5, 7], [3, 6, 5]]), axis=0))
print('Dtype=float (axis: 0): ', tf.reduce_mean(tf.constant([[2, 5, 7], [3, 6, 5]], dtype=tf.float64), axis=0))

print('最大值')
print('Axis0:', tf.reduce_max(tf.constant([[3, 4, 6], [6, 7, 0]]), axis=0))
print('Axis1:', tf.reduce_max(tf.constant([[3, 4, 6], [6, 7, 0]]), axis=1))
print('no axis:', tf.reduce_max(tf.constant([[3, 4, 6], [6, 7, 0]])))
print('最小值一样')

# --------------------------------------------------------

print('索引')
# tf.argmax()
# tf.argmin()
# 当轴未给定时, 默认axis为0
print('max:', tf.argmax(tf.constant([[658656, 2451, 9], [451, 646156, 56564], [56, 14564, 145646]])))
print('min:', tf.argmin(tf.constant([[658656, 2451, 9], [451, 646156, 56564], [56, 14564, 145646]])))

# --------------------------------------------------------

# 解析算法

# --------------------------------------------------------

# 一元线性回归
# y = wx + b
# w: 权重
# b: 偏置质

# 代价/损失函数
print(open('Algorithm/Loss.png'))