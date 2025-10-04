import os
import asyncio

try:
    from context_agent import ContextAgent
    from commentor_agent import CommentorAgent
    from review_posting_agent import ReviewAndPostingAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.groq import Groq
    from llama_index.llms.ollama import Ollama
    from llama_index.core.workflow import Context
    from llama_index.core.agent.workflow import AgentWorkflow
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.prompts import RichPromptTemplate
    from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
except ImportError as e:
    print(f"Error: Missing required package. Please install llama-index packages:")
    print("  pip install llama-index-core llama-index-llms-ollama")
    raise

from dotenv import load_dotenv
load_dotenv()

groq = Groq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

def build_agent_workflow():
    """Create a workflow to orchestrate agents"""
    context_agent = ContextAgent(groq).build_agent()
    commentor_agent = CommentorAgent(groq).build_agent()
    review_agent = ReviewAndPostingAgent(groq).build_agent()

    # Define steps for the workflow: ContextAgent runs first, then feeds output to CommentorAgent
    workflow = AgentWorkflow(
        agents=[context_agent, commentor_agent, review_agent],
        root_agent=review_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review_comment": "",
        },
    )
    return workflow

async def main():
    workflow = build_agent_workflow()
    Context(workflow)
    pr_number = os.getenv("PR_NUMBER")
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
