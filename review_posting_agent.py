import os
from github import Github, Auth
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

load_dotenv()

class ReviewAndPostingAgent:
    def __init__(self, llm=None):
        self.llm = llm or Ollama(model="gpt-oss", base_url=os.getenv("OLLAMA_BASE_URL"))

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        self.git = Github(auth=Auth.Token(github_token))
        self.repo = self.git.get_repo(os.getenv("REPOSITORY"))

    async def add_final_review_to_state(self, ctx: Context, final_review: str):
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["final_review_comment"] = final_review
        return "Final review ready to post."

    def post_final_review_to_github(self, final_review_comment: str) -> dict:
        pr = self.repo.get_pull(int(os.getenv("PR_NUMBER")))
        commit = self.repo.get_commit(pr.head.sha)
        review = pr.create_review(commit=commit, body=final_review_comment, event="COMMENT")
        return {"review_id": review.id, "state": review.state, "html_url": getattr(review, "html_url", None)}

    def build_agent(self) -> FunctionAgent:
        post_review_tool = FunctionTool.from_defaults(
            fn=self.post_final_review_to_github,
            name="post_final_review_to_github",
            description="Post final review comment to GitHub. Required: final_review_comment (str)"
        )

        system_prompt = (
            "Do not answer directly."
            "You are the Review and Posting agent. You must use the CommentorAgent to create a draft review comment."
            "Once a draft review is generated, you need to run a final check and post it to GitHub."
            "- The review must: \n"
            "- Be a ~100-200 word review in markdown format. \n"
            "- Specify what is good about the PR: \n"
            "- Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
            "If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
            "You MUST save final review to state and post the review to GitHub."
        )

        return FunctionAgent(
            llm=self.llm,
            name="ReviewAndPostingAgent",
            system_prompt=system_prompt,
            description="Posts the final review comment to GitHub.",
            tools=[post_review_tool, self.add_final_review_to_state],
            can_handoff_to=["CommentorAgent"]
        )
