import os
import asyncio
import logging
import traceback
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm
from .utils import get_default_parking_fields

# Dynamically extract fields from data_type.json
DEFAULT_PARKING_FIELDS = get_default_parking_fields()

AGENT_PROMPT_COMMON_RULES = """
- Always respond in the user's language. Detect the user's language from their last message and answer in that language.
- Use a friendly and approachable tone, as if you are chatting with a friend.
- Use emojis in your answers to make the conversation more fun and friendly! 🚗🅿️✨
"""

class UserFriendlyToolError(Exception):
    pass

async def get_param_error_message_ai(user_text, llm_agent, missing_params=None):
    if missing_params:
        param_str = ', '.join(missing_params)
        prompt = (
            f"The user tried to use a tool, but did not provide required information: {param_str}. "
            "Please generate a short, friendly, and clear error message in the user's language, "
            "explaining that the following information is missing: "
            f"{param_str}. "
            "Do NOT show any JSON, code example, or internal structure. "
            "Just mention the missing item names in a simple, user-friendly way. "
            "Do not mention internal parameter names or technical details. "
            f"User's last message: {user_text}"
        )
    else:
        prompt = (
            "The user tried to use a tool, but did not provide all required information. "
            "Please generate a short, friendly, and clear error message in the user's language, "
            "explaining that some required information is missing for the request, but do NOT show any JSON or code example. "
            "Do not mention internal parameter names. "
            f"User's last message: {user_text}"
        )
    response = await llm_agent.generate(prompt)
    return response.text if hasattr(response, 'text') else str(response)

async def ensure_required_params_callback(tool, args, tool_context):
    logging.info(f"[TOOL GUARDRAIL] Called ensure_required_params_callback with tool={tool}, args={args}, tool_context={tool_context}")
    try:
        required_params = getattr(tool, 'required', []) or []
        missing_params = [p for p in required_params if p not in args or args[p] in (None, "")]
        if missing_params:
            if "queryBody" in missing_params:
                user_message_en = (
                    "Your search is missing the required 'queryBody' parameter. "
                    "Please provide more details about your search (e.g., location, price, facility, space, etc.) so I can help you better!"
                )
                user_text = getattr(tool_context, "user_input", None)
                llm_agent = getattr(tool_context, "llm_agent", None)
                if llm_agent and user_text:
                    prompt = (
                        f"{AGENT_PROMPT_COMMON_RULES}\n"
                        f"Translate the following message into the user's language, matching the tone and style of the user's last message.\n"
                        f"User's last message: {user_text}\n"
                        f"Message: {user_message_en}"
                    )
                    try:
                        response = await llm_agent.generate(prompt)
                        user_message = response.text if hasattr(response, 'text') else str(response)
                    except Exception as e:
                        logging.error(f"[TOOL GUARDRAIL] LLM translation failed: {e}")
                        user_message = user_message_en
                else:
                    user_message = user_message_en
                return {
                    "status": "error",
                    "error_message": user_message
                }
            return {
                "status": "error",
                "error_message": f"Required information ({', '.join(missing_params)}) is missing. Please provide more details!"
            }
        logging.info("[TOOL GUARDRAIL] All required params present. Tool execution allowed.")
        return None
    except Exception as e:
        logging.error(f"[TOOL GUARDRAIL] Exception in ensure_required_params_callback: {e}")
        logging.error(traceback.format_exc())
        return {"status": "error", "error_message": f"Exception occurred during parameter check: {e}"}

async def create_agent():
    username = os.getenv("ES_USERNAME")
    password = os.getenv("ES_PASSWORD")
    es_url = os.getenv("ES_URL")
    tools = []
    exit_stack = None

    # Logging configuration 
    logging.basicConfig(level=logging.INFO)

    try:
        tools, exit_stack = await asyncio.wait_for(
            MCPToolset.from_server(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=["-y",
                        "@elastic/mcp-server-elasticsearch",
                    ],
                    env={
                        "ES_URL": es_url,
                        "ES_USERNAME": username,
                        "ES_PASSWORD": password
                    }
                )
            ),
            timeout=100
        )
        logging.info(f"Received {len(tools)} tools from the MCP server.")
    except asyncio.TimeoutError:
        logging.error("[Warning] MCP server connection was not completed within 10 seconds. Creating the agent without MCP tools.")
        logging.error(f"[MCP Connection] ES_URL: {es_url}, ES_USERNAME: {username}")
    except Exception as e:
        logging.error("[Warning] Error occurred while connecting to the MCP server. Creating the agent without MCP tools.")
        logging.error(f"[MCP Connection] ES_URL: {es_url}, ES_USERNAME: {username}")
        logging.error(f"[MCP Connection] Exception: {e}")
        logging.error(traceback.format_exc())

    fields_str = ", ".join(DEFAULT_PARKING_FIELDS)

    agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name='mcp_agent',
        instruction = f"""
            You are an agent that can use the Elasticsearch MCP server's search tool.

            {AGENT_PROMPT_COMMON_RULES}

            **All data queries must use only the \"parking\" index. Never use any other index.**

            **IMPORTANT: When calling the MCP search tool, you MUST always include the 'queryBody' parameter.**
            If 'queryBody' is missing, the tool will not work and an error will occur.
            Always construct and include a valid 'queryBody' object in your tool calls.

            When generating a queryBody, always include or prioritize the following fields by default, unless the user specifies otherwise:
            {fields_str}

            If the user's request is ambiguous, use these fields as the default set for your queryBody.

            - For text fields, use match queries.
            - For keyword fields, use term queries.
            - For boolean fields, use term queries with true/false.
            - For long/float fields, use range or term queries.
            - For date fields, use range queries.
            - For nested fields, use the nested query structure.

            Examples:
            {{
                "queryBody": {{
                    "query": {{
                        "match": {{
                            "address": "search term"
                        }}
                    }}
                }}
            }}

            Nested field example:
            {{
                "queryBody": {{
                    "query": {{
                        "nested": {{
                            "path": "spaces",
                            "query": {{
                                "term": {{ "spaces.isVisible": true }}
                            }}
                        }}
                    }}
                }}
            }}

            **Instructions:**
            - Always use only the \"parking\" index for all queries.
            - Use field names and types as defined above.
            - If you are unsure about a field, check the DEFAULT_PARKING_FIELDS list before constructing the query.

            Have fun helping the user! 😄
            """,
        tools=tools,
        before_tool_callback=ensure_required_params_callback,  # Register async callback directly
    )
    return agent, exit_stack

# Safe shutdown function
async def safe_aclose_exit_stack(exit_stack):
    if exit_stack is not None:
        try:
            await exit_stack.aclose()
            logging.info("[Shutdown] exit_stack closed successfully.")
        except Exception as e:
            logging.error(f"[Shutdown] Error while closing exit_stack: {e}")
            logging.error(traceback.format_exc())
    else:
        logging.info("[Shutdown] exit_stack is None, nothing to close.")

root_agent = create_agent()