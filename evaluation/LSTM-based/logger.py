# coding: utf-8
# Name:     logger2file
# Author:   tyler
# Data:     2022/4/19
import logging

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")       # 设置输出格式
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)                                                   # 输出日志的信息

    console_handler = logging.StreamHandler()                                       # 输出到控制台
    console_handler.setFormatter(log_format)                                        # 设置输出格式
    logger.handlers = [console_handler]                                             # 添加到logger对象中

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)                                # 文件处理器
        file_handler.setLevel(log_file_level)                                       # logging.NOTSET--全部打印
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    log_file = r"log/log.txt"
    logger = init_logger(log_file)

