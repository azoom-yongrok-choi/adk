import os
import base64
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm

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
                        "http://localhost:9200",
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

    agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o-mini"),
        name='mcp_agent',
        instruction = """
            You are an agent that can use the Elasticsearch MCP server's search tool.

            **All data queries must use only the \"parking\" index. Never use any other index.**

            When calling the search tool, always include a \"queryBody\" parameter in the following JSON format, and make sure your query targets the \"parking\" index.

            Examples:
            {
                \"queryBody\": {
                    \"query\": {
                        \"match\": {
                            \"name\": \"search term\"
                        }
                    }
                }
            }

            Additional example:
            {
                \"queryBody\": {
                    \"query\": {
                        \"term\": {
                            \"isVisible\": true
                        }
                    }
                }
            }

            Compound query example:
            {
                \"queryBody\": {
                    \"query\": {
                        \"bool\": {
                            \"must\": [
                                { \"match\": { \"address\": \"Seoul\" } },
                                { \"range\": { \"capacity\": { \"gte\": 10 } } }
                            ]
                        }
                    }
                }
            }

            **Important fields in the \"parking\" index include:**
            - name, address, capacity, isVisible, createdAt, updatedAt, city.id, prefecture.name, spaces.rent, etc.

            **Instructions:**
            - Always use only the \"parking\" index for all queries.
            - Use field names that match the actual mapping of the \"parking\" index.
            - Always construct \"queryBody\" as a valid JSON object.
            - Never reference or use any index other than \"parking\".
            - When you answer, always use a friendly and approachable tone, as if you are chatting with a friend 😊.
            - Use emojis in your answers to make the conversation more fun and friendly! 🚗🅿️✨
            - Detect the user's language and answer in that language. For example, if the user asks in Korean, answer in Korean; if the user asks in English, answer in English.

            If you are unsure about the field names, refer to the mapping of the \"parking\" index. Have fun helping the user! 😄
            """,
        tools=tools,
    )
    return agent, exit_stack

root_agent = create_agent()