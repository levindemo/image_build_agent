import os
import subprocess
import sqlite3
from typing import TypedDict, List, Optional, Dict
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# 状态定义
class AgentState(TypedDict):
    """智能体状态定义"""
    dockerfile_path: str  # Dockerfile 路径
    image_name: str  # 镜像名称
    build_result: Optional[str]  # 构建结果
    error_message: Optional[str]  # 错误信息
    retry_count: int  # 重试次数
    llm_suggestion: Optional[str]  # LLM 建议
    validation_result: Optional[str]  # 验证结果


# 数据库操作
class ErrorDatabase:
    """错误信息数据库"""

    def __init__(self, db_name: str = "docker_build_errors.db"):
        self.conn = sqlite3.connect(db_name)
        self._create_table()

    def _create_table(self):
        """创建错误记录表"""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS build_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_name TEXT,
            dockerfile_path TEXT,
            error_message TEXT
        )
        ''')
        self.conn.commit()

    def record_error(self, image_name: str, dockerfile_path: str, error_message: str):
        """记录错误信息"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO build_errors (image_name, dockerfile_path, error_message)
        VALUES (?, ?, ?)
        ''', (image_name, dockerfile_path, error_message))
        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


# Docker 操作工具
@tool
def build_docker_image(dockerfile_path: str, image_name: str) -> Dict[str, str]:
    """
    构建 Docker 镜像

    Args:
        dockerfile_path: Dockerfile 所在路径
        image_name: 要构建的镜像名称

    Returns:
        包含构建结果的字典
    """
    try:
        # 执行 docker build 命令
        result = subprocess.run(
            ["docker", "build", "-t", image_name, dockerfile_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'  # 指定编码

        )

        return {
            "status": "success",
            "message": f"镜像 {image_name} 构建成功",
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"镜像 {image_name} 构建失败",
            "error": e.stderr
        }


@tool
def start_and_validate_docker_image(image_name: str) -> Dict[str, str]:
    """
    启动 Docker 镜像并验证其状态

    Args:
        image_name: 要启动和验证的镜像名称

    Returns:
        包含验证结果的字典
    """
    try:
        # 启动容器
        container_id = subprocess.check_output(
            ["docker", "run", "-d", image_name],
            text=True,
            encoding='utf-8'  # 指定编码
        ).strip()

        # 检查容器状态
        status = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
            text=True,
            encoding='utf-8'  # 指定编码

        ).strip()

        # 如果容器正在运行，验证成功
        if status == "running":
            # 清理 - 停止并删除容器
            subprocess.run(["docker", "stop", container_id],
                           check=True,
                           encoding='utf-8'  # 指定编码
                           )
            subprocess.run(["docker", "rm", container_id],
                           check=True,
                           encoding='utf-8'  # 指定编码
                           )

            return {
                "status": "success",
                "message": f"镜像 {image_name} 启动并验证成功",
                "container_id": container_id
            }
        else:
            return {
                "status": "error",
                "message": f"镜像 {image_name} 启动但状态异常: {status}",
                "container_id": container_id
            }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"镜像 {image_name} 启动或验证失败",
            "error": e.stderr
        }


# LLM 错误处理
def get_llm_suggestion(llm: BaseLanguageModel, error_message: str, dockerfile_path: str) -> str:
    """
    获取 LLM 对 Docker 构建错误的建议

    Args:
        llm: 语言模型实例
        error_message: 错误信息
        dockerfile_path: Dockerfile 路径

    Returns:
        LLM 提供的解决方案建议
    """
    # 读取 Dockerfile 内容
    dockerfile_content = ""
    try:
        dockerfile_full_path = os.path.join(dockerfile_path, "Dockerfile")
        if os.path.exists(dockerfile_full_path):
            with open(dockerfile_full_path, "r") as f:
                dockerfile_content = f.read()
        else:
            dockerfile_content = f"Dockerfile 不存在于路径: {dockerfile_full_path}"
    except Exception as e:
        dockerfile_content = f"无法读取 Dockerfile: {str(e)}"

    # 构建提示
    prompt = ChatPromptTemplate.from_template("""
    我在构建 Docker 镜像时遇到了以下错误:

    {error_message}

    这是我的 Dockerfile 内容:

    {dockerfile_content}

    请分析这个错误并提供具体的解决方案，告诉我应该如何修改 Dockerfile 或构建过程来解决这个问题。
    """)

    # 调用 LLM
    chain = prompt | llm
    response = chain.invoke({
        "error_message": error_message,
        "dockerfile_content": dockerfile_content
    })

    return response.content


# 节点函数
def build_image_node(state: AgentState) -> AgentState:
    """构建镜像节点"""

    result = build_docker_image.invoke({
        "dockerfile_path": state["dockerfile_path"],
        "image_name": state["image_name"]
    })

    new_state = state.copy()
    new_state["build_result"] = result["status"]

    if result["status"] == "error":
        new_state["error_message"] = result["error"]
    else:
        new_state["error_message"] = None

    return new_state


def check_build_result_node(state: AgentState) -> str:
    """检查构建结果，决定下一步操作"""
    if state["build_result"] == "success":
        return "validate"
    elif state["retry_count"] < 1:
        return "retry"
    else:
        return "consult_llm"


def retry_build_node(state: AgentState) -> AgentState:
    """重试构建节点"""
    # 记录错误到数据库
    db = ErrorDatabase()
    db.record_error(
        state["image_name"],
        state["dockerfile_path"],
        state["error_message"]
    )
    db.close()

    # 重试构建

    result = build_docker_image.invoke({
        "dockerfile_path": state["dockerfile_path"],
        "image_name": state["image_name"]
    })
    new_state = state.copy()
    new_state["build_result"] = result["status"]
    new_state["retry_count"] += 1

    if result["status"] == "error":
        new_state["error_message"] = result["error"]
    else:
        new_state["error_message"] = None

    return new_state


def consult_llm_node(state: AgentState, llm: BaseLanguageModel) -> AgentState:
    """咨询 LLM 节点"""
    suggestion = get_llm_suggestion(
        llm,
        state["error_message"],
        state["dockerfile_path"]
    )

    new_state = state.copy()
    new_state["llm_suggestion"] = suggestion
    return new_state


def validate_image_node(state: AgentState) -> AgentState:
    """验证镜像节点"""
    result = start_and_validate_docker_image.invoke(
        {"image_name": state["image_name"]}
    )

    new_state = state.copy()
    new_state["validation_result"] = result["status"]
    return new_state


# 构建智能体
def create_docker_builder_agent(llm: BaseLanguageModel) -> StateGraph:
    """创建 Docker 构建智能体"""
    # 初始化状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("build", build_image_node)
    workflow.add_node("retry_build", retry_build_node)
    workflow.add_node("consult_llm", lambda state: consult_llm_node(state, llm))
    workflow.add_node("validate", validate_image_node)

    # 添加工具节点
    tools = [build_docker_image, start_and_validate_docker_image]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # 设置边缘
    workflow.set_entry_point("build")
    workflow.add_conditional_edges(
        "build",
        check_build_result_node,
        {
            "validate": "validate",
            "retry": "retry_build",
            "consult_llm": "consult_llm"
        }
    )
    workflow.add_conditional_edges(
        "retry_build",
        check_build_result_node,
        {
            "validate": "validate",
            "retry": END,  # 已经重试过一次，不再重试
            "consult_llm": "consult_llm"
        }
    )
    workflow.add_edge("consult_llm", END)
    workflow.add_edge("validate", END)

    # 添加内存检查点
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# 使用示例
if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Docker 镜像构建智能体')
    parser.add_argument('--dockerfile-path', type=str, default='example_dockerfile',
                        help='Dockerfile 所在目录路径')
    parser.add_argument('--image-name', type=str, default='my-test-image:latest',
                        help='要构建的镜像名称')
    parser.add_argument('--llm-provider', type=str, default='ollama',
                        choices=['openai', 'ollama', 'doubao'],
                        help='LLM 提供商 (openai, ollama, doubao)')
    parser.add_argument('--llm-model', type=str, default='deepseek-coder',
                        help='LLM 模型名称')

    args = parser.parse_args()

    # 配置 LLM
    llm = None
    if args.llm_provider == 'openai':
        from langchain_openai import ChatOpenAI

        # 确保已设置环境变量 OPENAI_API_KEY
        llm = ChatOpenAI(model_name=args.llm_model, temperature=0)
    elif args.llm_provider == 'ollama':
        from langchain_ollama import OllamaLLM

        llm = OllamaLLM(model=args.llm_model, temperature=0)

    elif args.llm_provider == 'doubao':
        from langchain_community.chat_models import DoubaoChat

        # 确保已设置环境变量 DOUBAO_API_KEY
        llm = DoubaoChat(model_name=args.llm_model, temperature=0)

    if not llm:
        raise ValueError("无法初始化 LLM，请检查配置")

    # 创建智能体
    agent = create_docker_builder_agent(llm)

    # 初始状态
    initial_state = {
        "dockerfile_path": args.dockerfile_path,
        "image_name": args.image_name,
        "build_result": None,
        "error_message": None,
        "retry_count": 0,
        "llm_suggestion": None,
        "validation_result": None
    }

    print(f"开始构建 Docker 镜像: {args.image_name}")
    print(f"Dockerfile 路径: {args.dockerfile_path}")
    print(f"使用 LLM: {args.llm_provider} - {args.llm_model}")
    print("----------------------------------------")

    # 运行智能体
    final_state = None
    config = {"configurable": {"thread_id": "1"}}
    for state in agent.stream(initial_state, config=config):
        for key, value in state.items():
            print(f"阶段: {key}")
            if key == "build" or key == "retry_build":
                print(f"输入: {value}")
                if value["build_result"] == "success":
                    print(f"结果: 构建成功")
                else:
                    print(f"结果: 构建失败")
                    print(f"错误信息: {value['error_message'][:500]}...")  # 只显示前500字符
            elif key == "validate":
                print(f"结果: {'验证成功' if value['validation_result'] == 'success' else '验证失败'}")
            elif key == "consult_llm":
                print(f"LLM 建议: {value['llm_suggestion'][:500]}...")  # 只显示前500字符
            print("----------------------------------------")
        final_state = state

    # 输出最终结果
    print("最终结果:")
    if final_state:
        last_node = next(reversed(final_state.keys()))
        last_state = final_state[last_node]

        if last_state["build_result"] == "success" and last_state["validation_result"] == "success":
            print(f"✅ 成功构建并验证镜像: {last_state['image_name']}")
        elif last_state["llm_suggestion"]:
            print(f"❌ 构建失败，LLM 建议: {last_state['llm_suggestion']}")
        else:
            print(f"❌ 构建失败: {last_state['error_message']}")
