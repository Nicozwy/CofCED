import logging
from logging import handlers

import os
from os.path import join as pjoin
import time
from datetime import datetime

date_str = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '')

ROOT_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\dataset\\logs"

filename = pjoin(ROOT_PROJ_PATH, f'{date_str}_AFC.log')
#
# logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                     level=logging.INFO,
#                     filename=pjoin(ROOT_PROJ_PATH, f'{date_str}_AFC.log'),
#                     filemode='a')

class SimpleLogger(object):

    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        #when=D，新生成的文件名上会带上时间
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
if __name__ == '__main__':
    log = SimpleLogger(filename, level='info', fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    log.logger.debug('debug')
    print('info', filename)
    log.logger.info("\nevaluating model on: %s %s", filename, "\n")
    log.logger.info('info %s', filename)
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    # Logger('error.log', level='error').logger.error('error')
