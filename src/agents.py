from smolagents import Tool, ToolCallingAgent
from smolagents.models import OpenAIModel
from smolagents import FinalAnswerTool, UserInputTool
from src.tools import Simple_Conversation_Tool, FAQTool
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Agent:

    def query_tool_agent(query: str) -> str:
        model = OpenAIModel(
            model_id="gpt-4o-mini",             # or "gpt-4o", "gpt-4.1", etc.
            api_key=OPENAI_API_KEY
        )

        # --- Register tools ---
        tools = [
                Simple_Conversation_Tool(api_key=OPENAI_API_KEY, model_id="gpt-4o-mini"),
                FAQTool(api_key=OPENAI_API_KEY, model_id="gpt-4o-mini", docx_path="general_faqs.docx"),
                FinalAnswerTool(),
                UserInputTool()
            ]

        INSTRUCTIONS = (
            "You are a tool-using customer support agent.\n"
            "Protocol for EVERY user message:\n"
            "1) Select exactly ONE domain tool (e.g., Simple_Conversation_Tool for greetings, "
            "   FAQTool for policy questions) and call it ONCE to produce a CANDIDATE_ANSWER.\n"
            "2) If the CANDIDATE_ANSWER fully addresses the user’s question:\n"
            "   - Call FinalAnswerTool with that answer and STOP.\n"
            "3) If the CANDIDATE_ANSWER is incomplete, unclear, or irrelevant:\n"
            "   - Call another domain tool that is most likely to improve the answer.\n"
            "   - OR, if clarification is absolutely required, call UserInputTool to ask the user.\n"
            "   - After refining, repeat step 2.\n"
            "4) If user question is beyond your capabilites answer, call UserInputTool "
            "   to request input from your customer support employee.\n"
            "Rules:\n"
            "- NEVER invent extra user messages or simulate a conversation with yourself.\n"
            "- Keep tool calls minimal — finalize as soon as you have a solid answer.\n"
            "- Always prefer short, clear, helpful answers.\n"
        )

        # --- Create the agent ---
        agent = ToolCallingAgent(
                model=model,
                tools=tools,
                instructions=INSTRUCTIONS,
                max_steps=5  # domain tool -> verify -> maybe one more hop -> verify -> finalize
            )

        # --- Run queries ---
        response = agent.run(query)

        return response

