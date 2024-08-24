# 使用官方的Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到工作目录
COPY . .

# 安装必要的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用程序的端口（假设FastAPI应用运行在5179端口）
EXPOSE 5179

# 运行应用程序
CMD ["uvicorn", "duck2api:app", "--host", "0.0.0.0", "--port", "5179"]
