# gunicorn_config.py

import os

# --- 服务器套接字 ---
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:5001')

# --- Worker 配置 ---
# !!! 关键改动: 更换 worker 类型为 gevent !!!
# gevent 使用协程，可以更好地处理 I/O 操作，并避免在 macOS 上由 fork() 引发的问题。
worker_class = 'gevent'

# 当使用 gevent 时，可以适当增加 worker 数量，因为它比 sync worker 更轻量。
# 但我们先从 1 开始，确保能成功运行。
workers = int(os.environ.get('GUNICORN_WORKERS', 1))

# --- 超时配置 ---
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))

# --- 日志 ---
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# --- 服务器钩子 ---
# 模型加载现在是按需进行的，每个 worker 独立加载和缓存模型。

print(f"Gunicorn config loaded. Worker Class: {worker_class}, Workers: {workers}")
