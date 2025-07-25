# -----------------------------------------------------------------------------
# 阶段 1: 使用官方 Python 基础镜像
# -----------------------------------------------------------------------------
# 使用 -slim 版本可以有效减小最终镜像的体积
FROM python:3.10-slim

# 设置一个环境变量，确保 Python 的输出能立即打印到终端，方便 docker logs 查看
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# 阶段 2: 设置工作目录并安装依赖
# -----------------------------------------------------------------------------
# 在容器内创建一个 /app 目录作为工作目录
WORKDIR /app

# 先只复制依赖文件，这样可以利用 Docker 的层缓存机制。
# 只要 requirements.txt 不变，就不需要重新执行 pip install，从而加快构建速度。
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# 阶段 3: 复制项目代码并设置启动命令
# -----------------------------------------------------------------------------
# 将当前目录下的所有文件复制到容器的 /app 目录中
COPY . .

# 声明容器内部服务监听的端口（与 gunicorn_config.py 中的端口保持一致）
# 这主要是一个文档性质的声明，方便使用者了解
EXPOSE 5001

# 容器启动时执行的命令
# Gunicorn 将会读取 gunicorn_config.py 的配置来启动
CMD ["gunicorn", "-c", "gunicorn_config.py", "web.app:app"]