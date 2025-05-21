# multi_tool_agent/main_async.py
import asyncio
from multi_tool_agent.agent import create_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types

async def async_main():
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
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    async for event in events_async:
        print(f"Event received: {event}")

    await exit_stack.aclose()

if __name__ == '__main__':
    asyncio.run(async_main())