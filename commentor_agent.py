import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent

load_dotenv()

class CommentorAgent:
    def __init__(self, llm=None):
        self.llm = llm or Ollama(model="gpt-oss", base_url=os.getenv("OLLAMA_BASE_URL"))

    async def save_draft_comment_to_state(self, ctx: Context, draft_comment: str):
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["review_comment"] = draft_comment
        return "Comment drafted."

    def build_agent(self) -> FunctionAgent:

        system_prompt = (
            "You are the commentor agent that writes draft review comments for pull requests as a human reviewer would.\n"
            "Do not answer directly. Use tools only."
            "You can ONLY gather information and save state via the tools provided. "
            "Ensure to do the following for a thorough review:"
            "- Request for the PR details, changed files, and any other repo files you may need from the ContextAgent."
            "- Once you have asked for all the needed information, write a good ~100-200 word review in markdown format detailing: \n"
            "- What is good about the PR? \n"
            "- Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n"
            "- If you need any additional details, you must hand off to the Context Agent. \n"
            "- You should directly address the author.So your comments should sound like: \n"
            "\"Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?\""
            "You MUST save your draft to state and hand off it to the ReviewAndPostingAgent."
        )

        return FunctionAgent(
            llm=self.llm,
            name="CommentorAgent",
            system_prompt=system_prompt,
            description="Drafts a pull review comment using context gathered.",
            tools=[self.save_draft_comment_to_state],
            can_handoff_to=["ReviewAndPostingAgent", "ContextAgent"]
        )
