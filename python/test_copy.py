import copy


original = [1, 2, [3, 4]]   # 包含不可变元素和可变子列表

ref = original  

shallow = copy.copy(original)   # 浅拷贝

deep = copy.deepcopy(original)  # 深拷贝

original[2].append(5)   # 修改原始数据中的可变子列表

original[0] = 100       # 修改原始数据的不可变元素


print("original =", original)
print("ref =", ref)
print("shallow =", shallow)
print("deep =", deep)

"""
输出：
original = [100, 2, [3, 4, 5]]
ref = [100, 2, [3, 4, 5]]   ,与original完全等价
shallow = [1, 2, [3, 4, 5]] ,列表地址仍然相同
deep = [1, 2, [3, 4]]       ,完全新的内容
"""