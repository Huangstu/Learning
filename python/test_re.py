import re

# 基本匹配 
pattern = r"abc"
text = "xyz abc def"
match = re.search(pattern, text)          # 搜索第一个匹配
print(match.group())                      # abc

# re.match 从字符串开头匹配
print(re.match(r"\d+", "123abc"))         # <re.Match object> (匹配123)
print(re.match(r"\d+", "abc123"))         # None


# 元字符 
# . 匹配任意字符（除换行）
print(re.search(r"a.c", "abc"))           # abc

# ^ 匹配开头
print(re.search(r"^abc", "abcxyz"))       # abc

# $ 匹配结尾
print(re.search(r"xyz$", "abcxyz"))       # xyz

# * 0次或多次
print(re.search(r"ab*", "a"))             # a

# + 1次或多次
print(re.search(r"ab+", "abbb"))          # abbb

# ? 0次或1次
print(re.search(r"ab?", "a"))             # a

# {n} 精确n次
print(re.search(r"a{3}", "aaabc"))        # aaa

# {n,} 至少n次
print(re.search(r"a{2,}", "aaaa"))        # aaaa

# {n,m} n到m次
print(re.search(r"a{2,4}", "aaaaa"))      # aaaa（贪婪）


# [] 字符集
print(re.search(r"[aeiou]", "hello"))     # e

# [^] 否定字符集
print(re.search(r"[^aeiou]", "hello"))    # h

# \d 数字, \D 非数字
print(re.search(r"\d+", "abc123def"))     # 123

# \w 单词字符（字母数字下划线）, \W 非单词
print(re.search(r"\w+", "hello_123!"))    # hello_123

# \s 空白符, \S 非空白
print(re.search(r"\s+", "a b"))           # ' '




# 分组与捕获
# 用 () 分组
m = re.search(r"(\d+)-(\d+)", "电话: 123-456")
print(m.group(0))                         # 123-456（整体）
print(m.group(1))                         # 123
print(m.group(2))                         # 456
print(m.groups())                         # ('123', '456')


# 命名分组 (?P<name>...)
m = re.search(r"(?P<area>\d+)-(?P<num>\d+)", "123-456")
print(m.group('area'))                    # 123

# 非捕获分组 (?:...)
m = re.search(r"(?:abc)+", "abcabc")      # 不捕获，但仍匹配
print(m.group())                          # abcabc



# re模块函数
text = "apple, banana, cherry, date"

# findall: 返回所有匹配列表
print(re.findall(r"\b\w+\b", text))       # ['apple', 'banana', 'cherry', 'date']
                                          # \b 表示单词边界
# finditer: 返回迭代器，每个元素是match对象
for m in re.finditer(r"\b\w+\b", text):
    print(m.group(), end=' ')             # apple banana cherry date
print()

# sub: 替换
new_text = re.sub(r"banana", "orange", text)
print(new_text)                           # apple, orange, cherry, date

# subn: 替换并返回替换次数
new_text, n = re.subn(r"a", "X", text)    # 替换所有'a'为'X'
print(new_text, n)                        # Xpple, bXnXnX, cherry, dXte 4

# split: 按模式分割
print(re.split(r",\s*", text))            # ['apple', 'banana', 'cherry', 'date']



# 编译正则表达式
pat = re.compile(r"\b\w{5}\b")            # 匹配5个字母的单词
print(pat.findall(text))                  # ['apple']



# 
# 匹配邮箱
email_pat = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
print(re.search(email_pat, "my email is test@example.com").group())  # test@example.com

# 匹配手机号
phone_pat = r"1[3-9]\d{9}"
print(re.search(phone_pat, "call 13812345678 now").group())          # 13812345678

# 贪婪与非贪婪
print(re.search(r"a.*b", "a111b222b").group())   # a111b222b（贪婪，尽量多匹配）
print(re.search(r"a.*?b", "a111b222b").group())  # a111b（非贪婪，?使*变成懒惰）

# flags
# re.IGNORECASE (re.I) 忽略大小写
print(re.search(r"hello", "HELLO", re.I))        # HELLO

# re.DOTALL 使 . 匹配换行
print(re.search(r"a.c", "a\nc", re.DOTALL))      # a\nc

# re.MULTILINE 使 ^ $ 匹配每行
text_multi = "line1\nline2\nline3"
print(re.findall(r"^line\d", text_multi, re.M))  # ['line1', 'line2', 'line3']



# 转义
print(re.search(r"\.", "abc.def"))               # '.'

