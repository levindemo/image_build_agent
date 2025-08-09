from flask import Flask, request, jsonify
import requests
import logging
from datetime import datetime
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docker_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Ollama服务地址（默认本地地址）
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
# 使用的模型名称
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek")


def analyze_docker_log(log_content):
    """
    调用DeepSeek模型分析Docker构建日志
    """
    try:
        # 构建提示词，指导模型进行Docker日志分析
        prompt = f"""
        你是一位Docker专家，需要分析以下Docker构建日志，找出其中的错误，并提供具体的修改建议。

        请按照以下结构进行分析：
        1. 错误总结：简要描述发现的主要错误
        2. 错误位置：指出错误发生的大致位置
        3. 错误原因：解释错误发生的可能原因
        4. 修改建议：提供具体的、可操作的修改步骤

        Docker构建日志：
        {log_content}
        """

        # 调用Ollama API
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=60  # 设置超时时间
        )

        if response.status_code == 200:
            result = response.json()
            return result["message"]["content"]
        else:
            logger.error(f"Ollama API请求失败，状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return f"分析失败: Ollama服务返回错误状态码 {response.status_code}"

    except Exception as e:
        logger.error(f"分析日志时发生错误: {str(e)}", exc_info=True)
        return f"分析过程中发生错误: {str(e)}"


@app.route('/api/analyze-docker-log', methods=['POST'])
def analyze_log_endpoint():
    """
    分析Docker构建日志的API端点
    """
    try:
        data = request.json

        # 验证请求数据
        if not data or 'log_content' not in data:
            return jsonify({
                "success": False,
                "error": "请求中缺少log_content字段"
            }), 400

        log_content = data['log_content']

        # 记录请求
        logger.info(f"收到Docker日志分析请求，日志长度: {len(log_content)}字符")

        # 调用分析函数
        analysis_result = analyze_docker_log(log_content)

        # 返回结果
        return jsonify({
            "success": True,
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"API处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"服务器处理请求时发生错误: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        # 检查Ollama服务是否可用
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False
            },
            timeout=10
        )

        if response.status_code == 200:
            return jsonify({
                "status": "healthy",
                "service": "docker-log-analyzer",
                "ollama_available": True,
                "model": MODEL_NAME,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "degraded",
                "service": "docker-log-analyzer",
                "ollama_available": False,
                "error": f"Ollama服务返回状态码 {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }), 503

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "docker-log-analyzer",
            "ollama_available": False,
            "error": f"无法连接到Ollama服务: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 503


if __name__ == '__main__':
    # 可以通过环境变量配置端口和调试模式
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    logger.info(f"启动Docker日志分析API服务，端口: {port}, 调试模式: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
