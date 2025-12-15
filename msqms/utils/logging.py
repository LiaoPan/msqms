# -*- coding: utf-8 -*-
"""Logging"""
from loguru import logger

# config log format
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
# logger.add("logs/my_log.log", rotation="500 MB", level="INFO", format=log_format)
#remove auto attached logger
logger.remove()
# add console ouput
logger.add(sink=lambda message: print(message, end=""), level="DEBUG", format=log_format)

# export the configured logger
clogger = logger

# define inline print functions with `verobse` parameter.
# verbose = False, not print log info.
slog = lambda key, value, verbose=True: print("{}:{}".format(key, value)) if verbose else None


