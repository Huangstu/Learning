def test_zip():
    names = ["Alice", "Bob", "Charlie"]
    scores = [85, 92, 78]
    ages = [20, 21, 19,22]

    #输入：多个可迭代对象（如列表、元组、字符串等） 
    #输出：一个迭代器，每次迭代返回一个元组，包含各个输入对象中对应位置的元素
    #长度：等于最短输入对象的长度（多余的元素会被截掉）
    zipped = zip(names, scores)
    print(list(zipped))  # [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
    
    zipped_all = zip(names, scores, ages)
    for name, score, age in zipped_all:
        print(f"{name}: {score}分, {age}岁")  # Alice: 85分, 20岁
                                             # Bob: 92分, 21岁
                                             # Charlie: 78分, 19岁
    
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    print([x + y for x, y in zip(list1, list2)])  # [5, 7, 9]
    
    #转字典
    dict1 = dict(zip(names, scores))
    print(dict1)  # {'Alice': 85, 'Bob': 92, 'Charlie': 78}
    
    print()

def test_lambda():
    square = lambda x: x ** 2
    print(square(5))  # 25
    
    add = lambda a, b: a + b
    print(add(3, 7))  # 10
    
    students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
    students.sort(key=lambda x: x[1]) #根据第二个元素排序
    print(students)  # [('Charlie', 78), ('Alice', 85), ('Bob', 92)]
    students.sort(key=lambda x: x[1], reverse=True)
    print(students)  # [('Bob', 92), ('Alice', 85), ('Charlie', 78)]
    
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    even = list(filter(lambda x: x % 2 == 0, nums)) #filter 本身返回的是一个迭代器
    #even = [x for x in nums if x % 2 == 0] 
    print(even)  # [2, 4, 6, 8, 10]
    
    print()

def test_map():
    nums = [1, 2, 3, 4, 5]
    
    squared = list(map(lambda x: x ** 2, nums))
    print(squared)  # [1, 4, 9, 16, 25]
    
    strs = ["1", "2", "3"]
    ints = list(map(int, strs))
    print(ints)  # [1, 2, 3]
    
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    added = list(map(lambda x, y: x + y, list1, list2))
    print(added)  # [5, 7, 9]
    
    names = ["alice", "bob", "charlie"]
    capitalized = list(map(lambda s: s.capitalize(), names))
    print(capitalized)  # ['Alice', 'Bob', 'Charlie']
    
    print()

def test_combined():
    students = [
        {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 92},
        {"name": "Charlie", "score": 78},
        {"name": "David", "score": 88}
    ]
    
    passed = list(filter(lambda s: s["score"] >= 80, students))
    print("及格学生:")
    for s in passed:
        print(f"  {s['name']}: {s['score']}分")  # Alice: 85分
                                                 # Bob: 92分
                                                 # David: 88分
    
    names = list(map(lambda s: s["name"], students))
    scores = list(map(lambda s: s["score"], students))
    print(f"姓名: {names}")    # 姓名: ['Alice', 'Bob', 'Charlie', 'David']
    print(f"分数: {scores}")   # 分数: [85, 92, 78, 88]
    
    name_score = dict(zip(names, scores))
    print(f"字典: {name_score}")  # 字典: {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 88}
    
    good_students = list(map(
        lambda s: f"{s['name']}({s['score']})", 
        filter(lambda s: s["score"] >= 85, students)
    ))
    print(f"优秀学生: {good_students}")  # 优秀学生: ['Alice(85)', 'Bob(92)', 'David(88)']
    
    print()

if __name__ == "__main__":
    test_zip()
    test_lambda()
    test_map()
    test_combined()