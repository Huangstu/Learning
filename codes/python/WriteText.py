class WriteText:
    def __init__(self, name="我", filename="study_log.txt"):
        self.name = name
        self.filename = filename

    def write(self, action, detail=""):
        with open(self.filename, "a") as f:
            if detail:
                f.write(f"{self.name} {action}：{detail}\n")
            else:
                f.write(f"{self.name} {action}\n")

    def read(self):
        try:
            with open(self.filename, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "日志不存在。"

    def set_name(self, new_name):
        self.name = new_name
    def get_name(self):
        return self.name  

if __name__ == "__main__":
    log = WriteText()
    log.write("开始学习", "Python类")
    log.write("复习了读写文件")
    log.set_name("小明")
    log.write("完成了练习", "异常处理")
    print(log.read())