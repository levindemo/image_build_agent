# Docker 构建智能体

这是一个基于 LangGraph 和 LangChain 的智能体，用于自动构建 Docker 镜像并处理构建过程中出现的问题。

## 功能特点

1. 自动构建 Docker 镜像
2. 构建失败时自动记录错误到数据库并重试
3. 多次失败后咨询 LLM 获取解决方案
4. 构建成功后自动验证镜像运行状态

## 安装要求

- Python 3.8+
- Docker 已安装并正常运行
- （可选）OpenAI API 密钥（如使用 OpenAI 模型）
- （可选）豆包 API 密钥（如使用豆包模型）
- （可选）Ollama 已安装并下载相应模型（如使用本地模型）

## 安装步骤

1. 克隆或下载项目代码
2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```
3. （可选）设置环境变量配置 API 密钥：
   ```
   # 对于 OpenAI
   export OPENAI_API_KEY="你的API密钥"
   
   # 对于豆包
   export DOUBAO_API_KEY="你的API密钥"
   ```

## 使用方法

基本用法：python docker_builder_agent.py --dockerfile-path /path/to/dockerfile --image-name my-image:latest
指定 LLM 提供商和模型：# 使用 OpenAI 的 gpt-3.5-turbo
python docker_builder_agent.py --llm-provider openai --llm-model gpt-3.5-turbo

# 使用 Ollama 的 deepseek-coder
python docker_builder_agent.py --llm-provider ollama --llm-model deepseek-coder
## 项目结构

- `docker_builder_agent.py`: 智能体核心代码
- `requirements.txt`: 项目依赖
- `example_dockerfile/`: 示例 Dockerfile 及其应用代码
- `docker_build_errors.db`: 自动生成的错误记录数据库（SQLite）

## 工作流程

1. 智能体尝试构建指定的 Docker 镜像
2. 如果构建成功，智能体会启动镜像并验证其运行状态
3. 如果构建失败，智能体会：
   - 将错误信息记录到数据库
   - 自动重试一次构建
   - 如果再次失败，会咨询配置的 LLM 获取解决方案建议
    