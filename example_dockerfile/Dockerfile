# 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器的 /app 目录
COPY . /app

# 安装所需的包
RUN pip install --no-cache-dir flask

# 暴露端口
EXPOSE 5000

# 定义环境变量
ENV FLASK_APP=app.py

# 运行命令
CMD ["flask", "run", "--host=0.0.0.0"]
    