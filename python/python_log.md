##
### 


### Python Numpy

字符串、容器、Numpy

#### 字符串 (Strings)


```python
hello = 'hello'    
world = "world"    
print(hello)       # hello
print(len(hello))  # 5

hw = hello + ' ' + world          # 拼接
print(hw)                         # hello world

hw12 = '%s %s %d' % (hello, world, 12)  # sprintf 风格格式化
print(hw12)                               # hello world 12
```

##### 常用字符串方法

```python
s = "hello"
print(s.capitalize())           # Hello
print(s.upper())                # HELLO
print(s.rjust(7))               # "  hello"
print(s.center(7))              # " hello "
print(s.replace('l', '(ell)'))  # he(ell)(ell)o
print('  world '.strip())       # world
```

---

#### 容器 (Containers)

Python 内置了多种容器类型：列表、字典、集合、元组。

##### 列表 (Lists)

列表类似数组，但可动态调整大小，且元素类型可以不同。

```python
xs = [3, 1, 2]
print(xs, xs[2])     # [3, 1, 2] 2
print(xs[-1])        # 2（负索引从末尾开始）

xs[2] = 'foo'        # 可以放不同类型
print(xs)            # [3, 1, 'foo']

xs.append('bar')     # 末尾添加
print(xs)            # [3, 1, 'foo', 'bar']

x = xs.pop()         # 移除并返回最后一个元素
print(x, xs)         # bar [3, 1, 'foo']
```

##### 切片 (Slicing)

可以对列表进行切片

```python
nums = list(range(5))     # [0, 1, 2, 3, 4]
print(nums[2:4])          # [2, 3]（索引2到4，不含4）
print(nums[2:])           # [2, 3, 4]
print(nums[:2])           # [0, 1]
print(nums[:])            # [0, 1, 2, 3, 4]
print(nums[:-1])          # [0, 1, 2, 3]（负索引）

nums[2:4] = [8, 9]        # 切片赋值
print(nums)               # [0, 1, 8, 9, 4]
```

列表的循环

```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)         # cat, dog, monkey 各一行
```

使用 `enumerate` 同时获取索引和元素：

```python
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# #1: cat
# #2: dog
# #3: monkey
```

##### 列表推导式 (List Comprehensions)

```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # [0, 1, 4, 9, 16]
```

带条件：

```python
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # [0, 4, 16]
```

---

#### 字典 (Dictionaries)

存储键值对。

```python
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat'])                # cute
print('cat' in d)              # True

d['fish'] = 'wet'
print(d['fish'])               # wet

# print(d['monkey'])           # KeyError
print(d.get('monkey', 'N/A'))  # N/A（默认值）
print(d.get('fish', 'N/A'))    # wet

del d['fish']
print(d.get('fish', 'N/A'))    # N/A
```

循环遍历字典

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
```

使用 `items()` 同时获取键和值：

```python
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
```

##### 字典推导式

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # {0: 0, 2: 4, 4: 16}
```

---

#### 集合 (Sets)

无序、不重复元素的集合。

```python
animals = {'cat', 'dog'}
print('cat' in animals)   # True
print('fish' in animals)  # False

animals.add('fish')
print('fish' in animals)  # True
print(len(animals))       # 3

animals.add('cat')        # 已存在，无变化
print(len(animals))       # 3

animals.remove('cat')
print(len(animals))       # 2
```

 循环遍历集合（顺序不确定）

```python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# 输出顺序可能不同
```

##### 集合推导式

```python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # {0, 1, 2, 3, 4, 5}
```

---

#### 元组 (Tuples)

不可变的有序列表。可用作字典的键或集合的元素。


```python
t1 = (1, 2, 3)          # 标准写法
t2 = 4, 5, 6            # 可省略括号
t3 = (7,)               # 单元素必须加逗号
t4 = tuple([1, 2, 3])   # 从列表转换
```


##### 不可变
```python
t = (1, 2, 3)
# t[0] = 10             # 不允许修改
# t.append(4)           # 无添加方法
t = (4, 5, 6)           # 只能整体重新赋值
```

##### 可作为字典键或集合元素
```python
d = {(x, x+1): x for x in range(10)}  # 元组作为键
print(d[(5, 6)])        # 5


s = {(1, 2), (3, 4)}    # 集合元素
print((1, 2) in s)      # True
```



##### 元组解包

```python
a, b, c = (1, 2, 3)      # a=1, b=2, c=3

x, y = 10, 20            # 交换变量（经典用法）
x, y = y, x              # x=20, y=10

def get_point():
    return 5, 10         # 返回元组
x, y = get_point()       # x=5, y=10
```



##### 适用场景

用元组：
- 作为字典键或集合元素
- 表示固定数据（坐标、日期）
- 函数多返回值
- 数据不需要修改

用列表：
- 需要增删改的数据
- 数据数量不确定
- 需要排序等操作

##### 其他

```python
# 单元素元组必须有逗号
empty = ()              # 空元组
wrong = (1)             # 这是 int
correct = (1,)          # 正确写法

# 元组包含可变对象
t = ([1, 2], 3)         # 元组包含列表
# t[0] = [4, 5]         # 不能改变元素
t[0].append(6)          # 可以修改列表内容
```

#### Numpy 

Numpy 可以提供高性能多维数组对象及操作工具。

##### 数组创建

```python
import numpy as np

# 从列表创建
a = np.array([1, 2, 3])             # 一维数组
print(a.shape)                      # (3,)
b = np.array([[1,2,3],[4,5,6]])     # 二维数组
print(b.shape)                      # (2, 3)

# 特殊数组
c = np.zeros((2,2))                 # 全零数组
                                    # Prints "[[ 0.  0.]
                                    #          [ 0.  0.]]"
d = np.ones((1,2))                  # 全一数组
                                    # Prints "[[ 1.  1.]]"
e = np.full((2,2), 7)               # 常量数组
                                    # Prints "[[ 7.  7.]
                                    #          [ 7.  7.]]"
f = np.eye(2)                       # 单位矩阵
                                    # Prints "[[ 1.  0.]
                                    #          [ 0.  1.]]"
g = np.random.random((2,2))         # 随机数组
```

##### 数组索引

切片索引（返回视图，修改会影响原数组）：
```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]                      # 前两行，第1-2列
b[0, 0] = 77                        # 修改 b 会影响 a
print(a[0, 1])                      # Prints "77"
```


整数数组索引（返回新数组）：

```python
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# accessing the data in the middle row of the array.
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

#accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

```python
a = np.array([[1,2], [3,4], [5,6]])
print(a[[0, 1, 2], [0, 1, 0]])      # [1 4 5]
#同
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# 选取每行指定列
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
b = np.array([0, 2, 0, 1])          #指定列
print(a[np.arange(4), b])           # [1 6 7 11]

a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```


布尔数组索引：
```python
a = np.array([[1,2], [3,4], [5,6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

print(a[bool_idx])                  # [3 4 5 6]
print(a[a > 2])                     # 更简洁的写法
```

##### 数据类型

每个 Numpy 数组都是相同类型元素的网格。Numpy 提供了大量数值数据类型用于构建数组。创建数组时 Numpy 会尝试猜测数据类型，但数组构造函数通常也包含可选参数来显式指定数据类型。

```python
x = np.array([1, 2])
print(x.dtype)                          # int64

x = np.array([1.0, 2.0])
print(x.dtype)                          # float64

x = np.array([1, 2], dtype=np.int64)    # 指定类型
```

##### 数组运算

元素级运算：
```python
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x + y)                            
print(np.add(x, y))                     # 加法
# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]

print(x - y) 
print(np.subtract(x, y))  
# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]          
               
print(x * y)    
print(np.multiply(x, y))
# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]         
               
print(x / y)     
print(np.divide(x, y))    

print(np.sqrt(x))                       # 平方根
```

注意，与 MATLAB 不同，* 是元素级乘法，而不是矩阵乘法。我们使用 dot 函数来计算向量的内积、向量与矩阵相乘以及矩阵乘法。dot 既可作为 numpy 模块中的函数使用，也可作为数组对象的实例方法使用。

矩阵运算（使用 dot）：
```python
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))                       # 向量内积
print(x.dot(v))                       # 矩阵乘向量
print(x.dot(y))                       # 矩阵乘法
```

聚合函数：
```python
x = np.array([[1,2],[3,4]])
print(np.sum(x))                      # 全部求和：10
print(np.sum(x, axis=0))              # 按列求和：[4 6]
print(np.sum(x, axis=1))              # 按行求和：[3 7]
```

转置：
```python
x = np.array([[1,2], [3,4]])
print(x.T)                            # [[1 3] [2 4]]

v = np.array([1,2,3])
print(v.T)                            # 一维数组转置不变
```

##### 广播机制

广播允许不同形状的数组进行算术运算。

核心规则：
- 若数组秩不同，将低秩数组形状前面补1
- 各维度大小相同或其中一个为1时兼容
- 广播后形状为各维度最大值

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])

```

将向量 v 加到矩阵 x 的每一行等同于通过垂直堆叠 v 的多个副本来形成矩阵 vv，然后对 x 和 vv 进行逐元素求和。可以这样实现：

```python 
vv = np.tile(v, (4, 1))
print(vv)
y = x + vv
print(y)
```
NumPy 广播允许我们在不实际创建 v 的多个副本的情况下执行此计算。考虑使用广播的版本：

```python
# 向量加到矩阵每行（广播）
y = x + v                             # 自动广播

# 速度慢的写法
# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

```

这行代码 y = x + v 能够工作，即使 x 的形状为 (4, 3) 而 v 的形状为 (3,)，这是因为广播机制；这行代码的工作方式就像 v 实际上具有形状 (4, 3)，其中每一行都是 v 的副本，并且逐元素执行求和。

广播两个数组遵循以下规则：

- 如果数组不具有相同的秩，则将较低秩数组的形状前面补 1，直到两个形状具有相同的长度。

- 如果两个数组在某个维度上具有相同的大小，或者其中一个数组在该维度上大小为 1，则称这两个数组在该维度上是兼容的。

- 如果两个数组在所有维度上兼容，则可以广播。

- 广播后，每个数组的行为就好像它的形状等于两个输入数组形状的逐元素最大值。

- 在任何一个维度上，如果一个数组的大小为 1 而另一个数组的大小大于 1，则第一个数组的行为就好像它沿着该维度被复制了一样。


[numpy reference](https://numpy.org/doc/stable/reference/)


```python
# 外积
v = np.array([1,2,3])
w = np.array([4,5])
print(np.reshape(v, (3, 1)) * w)      # 列向量乘行向量

# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
#[[1],[2],[3]]*[4,5]

#[[1, 1]  [[4, 5]   [[1*4, 1*5]
#[2, 2] * [4, 5]  = [2*4, 2*5]
#[3, 3]]  [4, 5]]   [3*4, 3*5]]

# [[ 4  5]
#  [ 8 10]
#  [12 15]]

# 矩阵每列加向量
x = np.array([[1,2,3], [4,5,6]])
w = np.array([1,2])
print(x + np.reshape(w, (2, 1)))      # 或 (x.T + w).T

# 标量乘法
print(x * 2)                          # 标量广播
```

#### SciPy 扩展

SciPy 基于 Numpy，提供科学和工程应用函数。

##### 图像操作

```python
from scipy.misc import imread, imsave, imresize

img = imread('cat.jpg')               # 读取图像
print(img.shape)                      # (400, 248, 3)
img_tinted = img * [1, 0.95, 0.9]     # 调整颜色通道
img_tinted = imresize(img_tinted, (300, 300))  # 调整大小
imsave('cat_tinted.jpg', img_tinted)  # 保存图像
```

##### MATLAB 文件

```python
from scipy.io import loadmat, savemat
data = loadmat('file.mat')            # 读取 .mat 文件
savemat('file.mat', {'key': value})   # 保存为 .mat
```

##### 距离计算

```python
from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [2, 0]])
d = squareform(pdist(x, 'euclidean')) # 所有点对间距离
```

#### Matplotlib 绘图

##### 基本绘图

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine Curve')
plt.show()
```

多线图：
```python
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

##### 子图

```python
plt.subplot(2, 1, 1)                  # 2行1列第1个子图
plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)                  # 第2个子图
plt.plot(x, y_cos)
plt.title('Cosine')
plt.show()
```

##### 显示图像

```python
from scipy.misc import imread

img = imread('cat.jpg')
plt.imshow(np.uint8(img))             # 需要 uint8 类型
plt.show()
```
![cat](https://cs231n.github.io/assets/cat_tinted_imshow.png)
