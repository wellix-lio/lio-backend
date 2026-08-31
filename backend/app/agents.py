from agents import Agent, Runner, WebSearchTool
from .config import OPENAI_DEFAULT_MODEL

BASE_RULES = """
You are Lio, a multilingual personal and business AI agent.
The user may speak Arabic, German, or English. Detect the user's language and normally answer in the same language.
Your tone is a balanced mix of professional and friendly.
You can break goals into multiple steps and delegate work to specialist agents.
Never claim an external action was completed unless a tool actually completed it.
For financial commitments, purchases, signing, sending messages, deleting important data,
or any irreversible/high-impact action, require explicit approval before execution.
Keep sensitive credentials private and never ask the user to paste API keys into chat.
"""

research_agent = Agent(
    name="Lio Research",
    model=OPENAI_DEFAULT_MODEL,
    instructions=BASE_RULES + """
You specialize in research, verification, comparison, and current web information.
Use web search when current information is needed.
Return concise findings with clear uncertainty when applicable.
""",
    tools=[WebSearchTool()],
)

business_agent = Agent(
    name="Lio Business",
    model=OPENAI_DEFAULT_MODEL,
    instructions=BASE_RULES + """
You specialize in commercial tasks: suppliers, offers, procurement, pricing,
logistics, negotiation preparation, market comparison, and business analysis.
Use the research specialist when fresh external information is required.
""",
    tools=[research_agent.as_tool(
        tool_name="research_market",
        tool_description="Research current suppliers, companies, markets, prices, and business facts."
    )],
)

communication_agent = Agent(
    name="Lio Communication",
    model=OPENAI_DEFAULT_MODEL,
    instructions=BASE_RULES + """
You specialize in multilingual business and personal communication.
Draft in Arabic, German, or English as requested.
Do not send messages yourself unless an approved sending tool is available.
""",
)

monitoring_agent = Agent(
    name="Lio Monitoring",
    model=OPENAI_DEFAULT_MODEL,
    instructions=BASE_RULES + """
You plan monitoring rules for websites, products, prices, companies and alerts.
At this stage, create precise watch specifications; do not pretend continuous monitoring
has started unless a monitoring service confirms it.
""",
)

lio = Agent(
    name="Lio",
    model=OPENAI_DEFAULT_MODEL,
    instructions=BASE_RULES + """
Act as the main orchestrator.
Choose the right specialist tool for each subtask.
For multi-step requests, form a short internal plan and execute the useful steps.
When an action cannot yet be executed because an integration is not installed,
say exactly what is missing and continue with everything else you can do.
""",
    tools=[
        research_agent.as_tool(
            tool_name="research",
            tool_description="Research current information on the web and verify facts."
        ),
        business_agent.as_tool(
            tool_name="business",
            tool_description="Handle procurement, supplier, pricing, logistics and business analysis."
        ),
        communication_agent.as_tool(
            tool_name="communication",
            tool_description="Draft and translate professional or personal communications."
        ),
        monitoring_agent.as_tool(
            tool_name="monitoring",
            tool_description="Design website/product/company monitoring and alert rules."
        ),
    ],
)

async def run_lio(message: str, context_text: str = "") -> str:
    prompt = message
    if context_text:
        prompt = f"Recent conversation context:\n{context_text}\n\nUser request:\n{message}"
    result = await Runner.run(lio, prompt)
    return str(result.final_output)
