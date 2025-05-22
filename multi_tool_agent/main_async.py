# multi_tool_agent/main_async.py
import asyncio
from multi_tool_agent.agent import create_agent, UserFriendlyToolError, safe_aclose_exit_stack
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
import requests
import os

def test_es_connectivity():
    url = os.getenv("ES_URL")
    if not url:
        print("[MCP Connectivity] ES_URL 환경변수가 설정되어 있지 않습니다.")
        return
    try:
        resp = requests.get(url, timeout=10, verify=True)
        print(f"[MCP Connectivity] Status: {resp.status_code}, Body: {resp.text[:200]}")
    except Exception as e:
        print(f"[MCP Connectivity] Connection failed: {e}")

async def async_main():
    test_es_connectivity()
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()
    session = session_service.create_session(
        state={}, app_name='mcp_app', user_id='user1'
    )

    agent, exit_stack = await create_agent()
    runner = Runner(
        app_name='mcp_app',
        agent=agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    query = "list files in the tests folder"
    content = types.Content(role='user', parts=[types.Part(text=query)])

    print("Running agent...")
    try:
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )
        async for event in events_async:
            print(f"Event received: {event}")
    except UserFriendlyToolError as e:
        print(f"[User Message] {str(e)}")
    except Exception as e:
        print(f"[System Error] {str(e)}")

    if exit_stack is not None:
        await safe_aclose_exit_stack(exit_stack)

if __name__ == '__main__':
    asyncio.run(async_main())