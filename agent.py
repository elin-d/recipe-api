"""
A standalone implementation for the "Context manager agent" stage.

This file provides:
  - GitHub helper functions: get_pr_details, get_file_contents, get_commit_details
  - Conversion of the functions into LlamaIndex FunctionTool instances
  - A function to build the ReAct-based ContextAgent
  - An async main() that runs the agent with event streaming (example usage)
"""

import os
import asyncio

from github import Github, Auth

try:
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

# Initialize GitHub client
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GITHUB_TOKEN environment variable is required")

git = Github(auth=Auth.Token(github_token))

full_repo_name = os.getenv("REPOSITORY")

repo = git.get_repo(full_repo_name)

def build_context_agent(llm=None):
    """Create a ReActAgent configured to gather context from GitHub repos."""

    if llm is None:
        llm = Ollama(
            model="gpt-oss",
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )

    async def add_context_to_state(ctx: Context, gathered_contexts: str):

        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["gathered_contexts"] = gathered_contexts
        return "Context gathered."

    def get_pr_details(pr_number: int) -> dict:
        """Fetch pull request details: author, title, body, diff_url, state, head_sha, commits"""
        pr = repo.get_pull(pr_number)
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

    def get_file_contents(file_path: str) -> str:
        """Fetch file content from the repository"""
        file_content = repo.get_contents(file_path)
        return file_content.decoded_content.decode("utf-8")

    def get_commit_details(head_sha: str) -> list[dict]:
        """Fetch commit details: changed files with stats and patches"""
        commit = repo.get_commit(head_sha)
        changed_files: list[dict] = []
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

    # Create tools with explicit names and descriptions
    pr_tool = FunctionTool.from_defaults(
        fn=get_pr_details,
        name="get_pr_details",
        description=(
            "Fetch pull request details including author, title, body, state, "
            "head SHA, commit SHAs, and changed files. "
            "Required parameter: pr_number (int) - the pull request number"
        )
    )
    file_tool = FunctionTool.from_defaults(
        fn=get_file_contents,
        name="get_file_contents",
        description=(
            "Retrieve file contents from the repository at a specific path. "
            "Required parameter: path (str) - the file path in the repository. "
            "Optional parameter: ref (str) - git reference (branch name or commit SHA)"
        )
    )
    commit_tool = FunctionTool.from_defaults(
        fn=get_commit_details,
        name="get_commit_details",
        description=(
            "Get commit details including all changed files with their status, additions, deletions, and patches. "
            "Required parameter: head_sha (str) - the commit SHA"
        )
    )

    system_prompt = (
        "You are the only context gathering agent. When gathering context, you MUST gather:\n"
        "- The details: author, title, body, diff_url, state, and head_sha; \n"
        "- Changed files; \n"
        "- Any requested for files; \n"
        "Once you gather the requested info, you MUST hand control back to the CommentorAgent."
    )

    agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers all the needed context from GitHub and save it to state.",
        tools=[pr_tool, file_tool, commit_tool, add_context_to_state],
        system_prompt=system_prompt,
        can_handoff_to=["CommentorAgent"]
    )

    return agent

def build_commentor_agent(llm=None):
    """Create a ReActAgent configured to generate draft PR comments."""
    if llm is None:
        llm = Ollama(
            model="gpt-oss",
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )

    async def save_draft_comment_to_state(ctx: Context, draft_comment: str):
        """Return draft so the workflow can store it without touching outer Context."""
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["review_comment"] = draft_comment
        return "Comment drafted."

    system_prompt = (
    "You are the commentor agent that writes draft review comments for pull requests as a human reviewer would.\n"
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

    agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        system_prompt=system_prompt,
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        tools=[save_draft_comment_to_state],
        can_handoff_to = ["ReviewAndPostingAgent", "ContextAgent"]
    )

    return agent

def build_posting_agent(llm=None):
    """Create a ReActAgent configured to post the draft PR comments."""
    if llm is None:
        llm = Ollama(
            model="gpt-oss",
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )

    async def add_final_review_to_state(ctx: Context, final_review: str):
        """Add the final review to the state under the final_review key."""
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["final_review_comment"] = final_review
        return "Final review ready and need to be posted to GitHub."

    def post_final_review_to_github(final_review_comment: str) -> dict:
        """Post the final review comment to GitHub on the given pull request.

        This fetches the PR and uses create_review() to post the comment as a review body.
        """
        pr = repo.get_pull(int(os.getenv("PR_NUMBER")))

        commit = repo.get_commit(pr.head.sha)
        review = pr.create_review(
            commit=commit,
            body=final_review_comment,
            event="COMMENT",
        )
        return {
            "review_id": review.id,
            "state": review.state,
            "html_url": getattr(review, "html_url", None),
        }

    system_prompt = (
    "You are the Review and Posting agent. You must use the CommentorAgent to create a draft review comment."
    "Once a draft review is generated, you need to run a final check and post it to GitHub."
    "- The review must: \n"
    "- Be a ~100-200 word review in markdown format. \n"
    "- Specify what is good about the PR: \n"
    "- Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
    "If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
    "You MUST save final review to state and post the review to GitHub."
    )

    post_review_tool = FunctionTool.from_defaults(
        fn=post_final_review_to_github,
        name="post_final_review_to_github",
        description=(
            "Post the final review comment to GitHub on a specified pull request. "
            "Required parameters: pr_number (int) and final_review_comment (str) - the comment body."
        )
    )

    agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        system_prompt=system_prompt,
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        tools=[post_review_tool, add_final_review_to_state],
        can_handoff_to = ["CommentorAgent"]
    )

    return agent

def build_agent_workflow():
    """Create a workflow to orchestrate the ContextAgent and CommentorAgent."""
    context_agent = build_context_agent(groq)
    commentor_agent = build_commentor_agent(groq)
    review_and_posting_agent = build_posting_agent(groq)

    # Define steps for the workflow: ContextAgent runs first, then feeds output to CommentorAgent
    workflow = AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_posting_agent],
        root_agent=review_and_posting_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review_comment": "",
        },
    )
    return workflow

async def main():
    global agent_context
    workflow = build_agent_workflow()
    agent_context = Context(workflow)
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
    git.close()
