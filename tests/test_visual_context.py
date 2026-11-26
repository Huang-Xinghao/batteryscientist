"""Visual payloads survive tool execution, session storage and SDK serialization."""
import asyncio
import base64
import copy
import io
import json

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from batteryscientist.agent.context import WorkingContext, project_messages, summarize_messages
from batteryscientist.agent.loop import AgentLoop, AgentState
from batteryscientist.agent.runtime.sessions import RoleSession
from batteryscientist.integrations.model import ModelClient, ModelResponse, ToolCall
from batteryscientist.tools.core import ToolSpec
from batteryscientist.tools.image_viewing import view_image


def visual_state(tmp_path):
    Image.new('RGB', (64, 48), '#35a7c9').save(tmp_path / 'preview.png')
    session = RoleSession(tmp_path, 'paper_writer')

    class PersistedLoop(AgentLoop):
        def _save_conversation(self, state):
            session.save({'role': 'paper_writer', 'channel': 'execution',
                          'messages': state.messages})

    loop = PersistedLoop(None, workspace=tmp_path, connect_mcp=False,
                         tools=(ToolSpec('view_image', 'Inspect pixels', view_image, True),))
    state = AgentState([{'role': 'system', 'content': 'Inspect the figure.'},
                        {'role': 'user', 'content': 'Check preview.png.'}], 'paper_figure')
    asyncio.run(loop._execute_tools(state, ModelResponse(tool_calls=(
        ToolCall('inspect-1', 'view_image', {'path': 'preview.png'}),))))
    state.messages = session.load()['messages']
    return loop, state


def test_pixels_reach_sdk_after_tool_and_session_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv('BATTERYSCIENTIST_STREAM', 'false')
    loop, state = visual_state(tmp_path)
    assert state.messages[-1]['content'][-1]['type'] == 'stored_image'
    durable = copy.deepcopy(state.messages)
    requests = []
    events = []
    loop.on_progress = lambda kind, data: events.append((kind, data))

    def receive(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'vision-test', 'object': 'chat.completion', 'created': 0, 'model': 'vision',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': 'Blue rectangle.'}}],
        })

    with httpx.Client(transport=httpx.MockTransport(receive)) as transport:
        with OpenAI(api_key='test', base_url='https://vision.example/v1', http_client=transport) as sdk:
            loop.model = ModelClient('vision', client=sdk)
            response = asyncio.run(loop._request_model(state, [], WorkingContext()))
    assert response.content == 'Blue rectangle.'
    assert next(data for kind, data in events if kind == 'context_prepared')['image_count'] == 1
    block = requests[0]['messages'][-1]['content'][-1]
    assert block['type'] == 'image_url'
    with Image.open(io.BytesIO(base64.b64decode(block['image_url']['url'].split(',', 1)[1]))) as image:
        assert image.size == (64, 48)
        assert image.getpixel((20, 20)) == (53, 167, 201)
    assert state.messages == durable


def test_history_and_summary_keep_visual_findings_without_pixel_payloads(tmp_path):
    loop, state = visual_state(tmp_path)
    finding = 'The blue rectangle is fully visible with no clipping.'
    state.messages.append({'role': 'assistant', 'content': finding})
    state.messages.append({'role': 'user', 'content': 'Prepare the delivery.'})
    projected = project_messages(state.messages, image_store=loop.image_store)
    assert projected[-2]['content'] == finding
    assert not any(b.get('type') in {'image_url', 'stored_image'}
                   for m in projected if isinstance(m.get('content'), list) for b in m['content'])
    inputs = []

    def summarize(**kwargs):
        inputs.append(kwargs['messages'])
        return ModelResponse(content=finding)

    summary = summarize_messages(ModelClient(responder=summarize), state.messages, 100000)
    assert finding in summary
    assert finding in json.dumps(inputs)
    assert 'data:image' not in json.dumps(inputs)


def test_missing_fresh_pixels_raise_instead_of_becoming_a_text_placeholder(tmp_path):
    loop, state = visual_state(tmp_path)
    for asset in loop.image_store.root.iterdir():
        asset.unlink()
    with pytest.raises(FileNotFoundError):
        WorkingContext().prepare(state, None, 100000, lambda: None,
                                 image_store=loop.image_store)
