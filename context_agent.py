import os
from github import Github, Auth
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent

load_dotenv()

class ContextAgent:
    def __init__(self, llm=None):
        self.llm = llm or Ollama(model="gpt-oss", base_url=os.getenv("OLLAMA_BASE_URL"))

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        self.git = Github(auth=Auth.Token(github_token))
        self.repo = self.git.get_repo(os.getenv("REPOSITORY"))

    async def add_context_to_state(self, ctx: Context, gathered_contexts: str):
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["gathered_contexts"] = gathered_contexts
        return "Context gathered."

    def get_pr_details(self, pr_number: int) -> dict:
        pr = self.repo.get_pull(pr_number)
        commit_SHAs = [c.sha for c in pr.get_commits()]
        return {
            'author': pr.user.login,
            'title': pr.title,
            'body': pr.body or '<Empty body>',
            'diff_url': pr.diff_url,
            'state': pr.state,
            'head_sha': pr.head.sha,
            'commit_SHAs': commit_SHAs,
        }

    def get_file_contents(self, file_path: str) -> str:
        file_content = self.repo.get_contents(file_path)
        return file_content.decoded_content.decode("utf-8")

    def get_commit_details(self, head_sha: str) -> list[dict]:
        commit = self.repo.get_commit(head_sha)
        changed_files = []
        for f in commit.files:
            changed_files.append({
                'filename': f.filename,
                'status': f.status,
                'additions': f.additions,
                'deletions': f.deletions,
                'changes': f.changes,
                'patch': f.patch,
            })
        return changed_files

    def build_agent(self) -> FunctionAgent:
        pr_tool = FunctionTool.from_defaults(
            fn=self.get_pr_details,
            name="get_pr_details",
            description=(
                "Fetch pull request details including author, title, body, state, "
                "head SHA, commit SHAs, and changed files. "
                "Required parameter: pr_number (int) - the pull request number"
            )
        )
        file_tool = FunctionTool.from_defaults(
            fn=self.get_file_contents,
            name="get_file_contents",
            description=(
                "Retrieve file contents from the repository at a specific path. "
                "Required parameter: path (str) - the file path in the repository. "
                "Optional parameter: ref (str) - git reference (branch name or commit SHA)"
            )
        )
        commit_tool = FunctionTool.from_defaults(
            fn=self.get_commit_details,
            name="get_commit_details",
            description=(
                "Get commit details including all changed files with their status, additions, deletions, and patches. "
                "Required parameter: head_sha (str) - the commit SHA"
            )
        )
        add_context_tool = self.add_context_to_state

        system_prompt = (
            "You must never answer directly."
            "You can ONLY gather information via the tools provided. "
            "When gathering context, you MUST gather:\n"
            "- The details: author, title, body, diff_url, state, and head_sha; \n"
            "- Changed files; \n"
            "- Any requested for files; \n"
            "Once you gather the requested info, you MUST hand control back to the CommentorAgent."
        )

        return FunctionAgent(
            llm=self.llm,
            name="ContextAgent",
            description="Gathers needed context from GitHub and saves it to state.",
            tools=[pr_tool, file_tool, commit_tool, add_context_tool],
            system_prompt=system_prompt,
            can_handoff_to=["CommentorAgent"]
        )
