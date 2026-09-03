##
### date0903


#### Python Numpy

字符串、容器

#### 字符串 (Strings)


```python
hello = 'hello'    # 单引号
world = "world"    # 双引号
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
d = {(x, x + 1): x for x in range(10)}  # 元组作为键
t = (5, 6)
print(type(t))        # <class 'tuple'>
print(d[t])           # 5
print(d[(1, 2)])      # 1
```

