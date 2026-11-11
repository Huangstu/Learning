
### Linux
#### 1. 文件系统导航与基础操作

##### 查看与切换目录
- `pwd` - 显示当前路径
- `ls [选项] [目录]` - 列出内容（`-l` 长格式，`-a` 隐藏文件，`-h` 人类可读）
- `cd [目录]` - 切换目录（`~` 家目录，`-` 上一个目录，`..` 上级）

##### 创建、复制、移动、删除
- `mkdir [-p] 目录` - 创建目录（`-p` 递归创建父目录）
- `rm [-rf] 文件/目录` - 删除（`-r` 递归，`-f` 强制）
- `cp [-r] 源 目标` - 复制（`-r` 复制目录）
- `mv 源 目标` - 移动或重命名


**创建学习项目目录**
```bash
mkdir -p ~/learn/{linux,python,pytorch}
cd ~/learn
touch linux/notes.md python/script.py pytorch/model.py
ls -R   # 递归显示目录树
```

---

#### 2. 文件内容查看与文本处理

##### 查看文件
- `cat 文件` - 全部显示（适合小文件）
- `less 文件` - 分页查看（空格翻页，`/` 搜索，`q` 退出）
- `head -n N 文件` / `tail -n N 文件` - 头/尾 N 行（`tail -f` 实时追踪）

##### 文本过滤与处理
- `grep [选项] 模式 文件` - 搜索（`-i` 忽略大小写，`-r` 递归，`-v` 反选，`-n` 行号）
- `sed 's/旧/新/g' 文件` - 替换文本
- `awk '{print $列号}'` - 提取列
- `sort` / `uniq` - 排序与去重（`uniq -c` 计数）
- `wc [-l]` - 统计行数/字数


**在 learn 项目中查找所有 Python 文件中的 print 语句**
```bash
grep -rn "print" ~/learn/python/ --include="*.py"
```

---


#### 3. 权限管理

##### 修改权限与所有者
- `chmod [模式] 文件` - 修改权限（数字法 `755`，符号法 `u+x`）
- `chown 用户:组 文件` - 修改所有者
- `chgrp 组 文件` - 修改属组

**给学习脚本添加执行权限**
```bash
chmod +x ~/learn/python/script.py
ls -l ~/learn/python/script.py   # 查看权限变化
```

---

#### 4. 重定向与管道

##### 重定向符号
- `>` - 覆盖输出到文件
- `>>` - 追加输出
- `2>` - 错误重定向
- `2>&1` - 合并错误到标准输出
- `<` - 输入重定向

##### 管道与组合
- `|` - 前一个输出作为后一个输入
- `&&` - 前一个成功才执行后一个
- `||` - 前一个失败才执行后一个
- `;` - 顺序执行
  

**统计学习目录中 Python 文件的行数**
```bash
find ~/learn -name "*.py" | xargs wc -l | sort -nr
```

---



#### 5. 进程管理

##### 查看与终止进程
- `ps aux` 或 `ps -ef` - 显示所有进程
- `top` / `htop` - 实时监控（交互式）
- `kill [-9] PID` - 终止进程（`-9` 强制）
- `killall 进程名` - 按名称终止
- `jobs` / `bg` / `fg` - 前后台作业控制
- `nohup 命令 &` - 后台运行且不受终端关闭影响


**后台运行一个长时间任务**
```bash
nohup python ~/learn/pytorch/train.py > train.log 2>&1 &
tail -f train.log   # 实时查看日志
```

---



#### 6. 磁盘与文件系统

##### 查看磁盘与目录大小
- `df -h` - 分区使用情况
- `du -sh 目录` - 目录总大小（`-a` 每个文件）

##### 其他磁盘操作
- `mount` / `umount` - 挂载/卸载
- `fdisk -l` - 查看分区表


**查看 learn 目录下各子目录占用**
```bash
du -sh ~/learn/* | sort -hr
```

---



#### 7. 网络常用命令

- `ip a` / `ifconfig` - 查看 IP
- `ping -c N 目标` - 测试连通性
- `curl` / `wget` - HTTP 请求与下载
- `netstat -tuln` / `ss -tuln` - 查看监听端口
- `scp` / `rsync` - 远程拷贝与同步
- `ssh 用户@主机` - 远程登录

**从远程服务器同步学习资料到本地**
```bash
rsync -av user@server:/home/user/learn/ ~/learn/
```

---



#### 8. 压缩与归档

- `tar -cvf 归档.tar 文件...` - 创建 tar
- `tar -xvf 归档.tar` - 解包
- `tar -czvf 归档.tar.gz 文件...` - 创建并 gzip 压缩
- `tar -xzvf 归档.tar.gz` - 解压
- `gzip` / `gunzip` - 单独压缩
- `zip -r 归档.zip 目录` / `unzip 归档.zip` - zip 格式


**备份整个 learn 项目**
```bash
tar -czvf ~/learn_backup_$(date +%Y%m%d).tar.gz ~/learn/
```

---



#### 9. 查找与定位

- `find 路径 -name "模式"` - 按文件名查找（`-type f/d`，`-size`，`-exec`）
- `locate 文件名` - 基于数据库快速查找（先 `updatedb`）
- `which 命令` - 查找可执行文件路径
- `whereis 命令` - 查找二进制、源码和 man 页

**查找 learn 下所有 .md 文件并统计总大小**
```bash
find ~/learn -name "*.md" -exec du -ch {} + | grep total
```

---


#### 10. 环境变量与别名

- `export 变量=值` - 设置临时环境变量
- `echo $变量` - 查看
- `PATH=$PATH:/新路径` - 添加执行路径
- `alias 别名='命令'` - 创建临时别名
- `source ~/.bashrc` - 使配置生效
- 

**为学习目录创建快捷别名**
```bash
alias learn='cd ~/learn'
alias py='python3 ~/learn/python/script.py'
# 持久化：添加到 ~/.bashrc
```

---


#### 11. 软件包管理（Ubuntu）

- `apt update` - 更新源
- `apt upgrade` - 升级已安装
- `apt install 包名` - 安装
- `apt remove` / `purge` - 卸载
- `apt search 关键词` - 搜索
- `dpkg -i 包.deb` - 安装本地 deb


**安装 Python 和 PyTorch 依赖**
```bash
sudo apt update
sudo apt install python3-pip
pip3 install torch torchvision
```

---

#### 12. 系统信息与日志

- `uname -a` - 内核版本
- `uptime` - 运行时间与负载
- `free -h` - 内存使用
- `lscpu` - CPU 信息
- `dmesg | tail` - 内核日志
- `journalctl -xe` - systemd 日志

---

---

#### 13. 计划任务（cron）

- `crontab -e` - 编辑当前用户任务
- `crontab -l` - 列出
- `crontab -r` - 删除所有
- 格式：`分 时 日 月 周 命令`


**每天凌晨 2 点备份 learn 项目**
```bash
# 在 crontab -e 中添加：
0 2 * * * tar -czf ~/backups/learn_$(date +\%Y\%m\%d).tar.gz ~/learn
```

---


#### 14. 文本编辑器

- `vim` / `vi` - 模式编辑器（`i` 插入，`Esc` 退出，`:wq` 保存，`:q!` 强制退出）
- `nano` - 简易编辑器（`Ctrl+O` 保存，`Ctrl+X` 退出）


**用 vim 编辑学习笔记**
```bash
vim ~/learn/linux/notes.md
# 按 i 进入插入模式，编辑后按 Esc，输入 :wq 保存退出
```

---
