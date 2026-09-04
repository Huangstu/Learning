import pickle
import os


os.makedirs('picklefile', exist_ok=True)

# 基本数据类型序列化与反序列化
# pickle 支持：bool, int, float, complex, str, bytes, None
data_int = 42
data_str = "hello pickle"
data_list = [1, 2, 3]

with open('./picklefile/basic.pkl', 'wb') as f:
    pickle.dump(data_int, f)
    pickle.dump(data_str, f)
    pickle.dump(data_list, f)

with open('./picklefile/basic.pkl', 'rb') as f:
    loaded_int = pickle.load(f)
    loaded_str = pickle.load(f)
    loaded_list = pickle.load(f)

print(loaded_int)          # 42
print(loaded_str)          # hello pickle
print(loaded_list)         # [1, 2, 3]

# 容器类型
# 列表、元组、字典、集合、frozenset
container = {
    'a': [1, 2.5, (3+4j)],
    'b': {'x', 'y', 'z'},
    'c': frozenset([10, 20]),
    'd': None,
    'e': b'bytes data'
}
with open('./picklefile/container.pkl', 'wb') as f:
    pickle.dump(container, f)

with open('./picklefile/container.pkl', 'rb') as f:
    loaded_container = pickle.load(f)
print(loaded_container)
# {'a': [1, 2.5, (3+4j)], 'b': {'y', 'x', 'z'}, 'c': frozenset({10, 20}), 'd': None, 'e': b'bytes data'}

# 自定义类实例（类定义必须在作用域内）
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

p = Person('Alice', 30)
with open('./picklefile/person.pkl', 'wb') as f:
    pickle.dump(p, f)

with open('./picklefile/person.pkl', 'rb') as f:
    loaded_p = pickle.load(f)
print(loaded_p)            # Person(name='Alice', age=30)

# 混合存储 – 同一个容器内混合不同类型，推荐
mixed_dict = {
    'int': 100,
    'str': 'mixed',
    'list': [1, 2],
    'tuple': (3, 4),
    'dict': {'inner': 'ok'},
    'class': Person('Bob', 25)
}
with open('./picklefile/mixed.pkl', 'wb') as f:
    pickle.dump(mixed_dict, f)      # 只 dump 一次

with open('./picklefile/mixed.pkl', 'rb') as f:
    restored = pickle.load(f)       # 只 load 一次，恢复完整字典
print(restored)
# {'int': 100, 'str': 'mixed', 'list': [1, 2], 'tuple': (3, 4), 'dict': {'inner': 'ok'}, 'class': Person(name='Bob', age=25)}

# 多次 dump 与多次 load 的一一对应关系
with open('./picklefile/multi.pkl', 'wb') as f:
    pickle.dump(10, f)          # 根对象1
    pickle.dump([20, 30], f)    # 根对象2
    pickle.dump("third", f)     # 根对象3

with open('./picklefile/multi.pkl', 'rb') as f:
    obj1 = pickle.load(f)       # 对应第一次 dump
    obj2 = pickle.load(f)       # 对应第二次 dump
    obj3 = pickle.load(f)       # 对应第三次 dump
    # obj4 = pickle.load(f)     # EOFError（文件已结束）
print(obj1, obj2, obj3)         # 10 [20, 30] third

# 安全提示：仅加载自己保存的或来自可信方的数据。