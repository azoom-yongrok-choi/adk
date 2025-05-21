import os
import base64
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm

# Fields required for search (Location, Fee, Security, Space)
DEFAULT_PARKING_FIELDS = [
    # Location
    "address", "addressView", "location", "city", "prefecture", "region", "nearbyStations",
    # Fee
    "payment", "spaces.rent", "spaces.rentMin", "spaces.rentTaxClass", "referralFeeTotal", "storageDocument.issuingFee",
    # Security
    "securityFacilities", "spaces.facility",
    # Space
    "spaces", "capacity", "hasDivisionDrawing"
]

async def create_agent():
    username = os.getenv("ES_USERNAME")
    password = os.getenv("ES_PASSWORD")
    tools = []
    exit_stack = None

    try:
        tools, exit_stack = await asyncio.wait_for(
            MCPToolset.from_server(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=["-y",
                        "@elastic/mcp-server-elasticsearch",
                    ],
                    env={
                        "ES_API_KEY": os.getenv("ES_API_KEY"),
                        "ES_URL": os.getenv("ES_URL"),
                        "ES_USERNAME": username,
                        "ES_PASSWORD": password
                    }
                )
            ),
            timeout=20
        )
        print(f"Received {len(tools)} tools from the MCP server.")
    except asyncio.TimeoutError:
        print("[Warning] MCP server connection was not completed within 10 seconds. Creating the agent without MCP tools.")
    except Exception as e:
        print(f"[Warning] Error occurred while connecting to the MCP server: {e}\nCreating the agent without MCP tools.")

    fields_str = ", ".join(DEFAULT_PARKING_FIELDS)

    agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name='mcp_agent',
        instruction = f"""
            You are an agent that can use the Elasticsearch MCP server's search tool.

            **All data queries must use only the \"parking\" index. Never use any other index.**

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
                \"queryBody\": {{
                    \"query\": {{
                        \"match\": {{
                            \"address\": \"search term\"
                        }}
                    }}
                }}
            }}

            Nested field example:
            {{
                \"queryBody\": {{
                    \"query\": {{
                        \"nested\": {{
                            \"path\": \"spaces\",
                            \"query\": {{
                                \"term\": {{ \"spaces.isVisible\": true }}
                            }}
                        }}
                    }}
                }}
            }}

            **Instructions:**
            - Always use only the \"parking\" index for all queries.
            - Use field names and types as defined above.
            - If you are unsure about a field, check the DEFAULT_PARKING_FIELDS list before constructing the query.
            - When you answer, always use a friendly and approachable tone, as if you are chatting with a friend 😊.
            - Use emojis in your answers to make the conversation more fun and friendly! 🚗🅿️✨
            - Detect the user's language and answer in that language. For example, if the user asks in Korean, answer in Korean; if the user asks in English, answer in English.

            Have fun helping the user! 😄
            """,
        tools=tools,
    )
    return agent, exit_stack

root_agent = create_agent()