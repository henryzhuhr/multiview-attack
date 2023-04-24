import datetime


# 包含当前时间的log
logheader = lambda: "\033[01;32m[%s]\033[0m" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
