"""Model thinking text is captured, streamed separately and stored once per turn."""

from types import SimpleNamespace

from batteryscientist.integrations.model import ModelClient, _normalise_response, _stream_response


class FakeStream(list):
    closed = False

    def close(self):
        self.closed = True


def chunk(content=None, *, reasoning=None, field="reasoning_content", finish=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        setattr(delta, field, reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish)], usage=None
    )


def test_stream_separates_thinking_from_answer_and_flushes_thinking_first():
    events = []
    stream = FakeStream(
        [
            chunk(reasoning="先看数据划分，"),
            chunk(reasoning="再比较方法。"),
            chunk("结论：按电芯划分。"),
            chunk(finish="stop"),
        ]
    )
    response = _stream_response(
        stream,
        on_delta=lambda text: events.append(("answer", text)),
        on_reasoning=lambda text: events.append(("thinking", text)),
    )
    assert response.content == "结论：按电芯划分。"
    assert response.reasoning == "先看数据划分，再比较方法。"
    # All thinking reaches the page before the first answer text.
    assert events[0][0] == "thinking" and "".join(t for k, t in events if k == "thinking") == response.reasoning
    assert [k for k, _ in events].index("answer") > max(i for i, (k, _) in enumerate(events) if k == "thinking")
    assert stream.closed


def test_openrouter_style_reasoning_field_is_read_too():
    response = _stream_response(
        FakeStream([chunk(reasoning="思考", field="reasoning"), chunk("答案", finish="stop")])
    )
    assert response.reasoning == "思考" and response.content == "答案"


def test_providers_without_thinking_return_empty_reasoning():
    response = _stream_response(FakeStream([chunk("只有答案"), chunk(finish="stop")]))
    assert response.reasoning == ""


def test_non_streamed_message_reasoning_is_kept():
    message = SimpleNamespace(content="答案", tool_calls=None, reasoning_content="非流式思考")
    value = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
    assert _normalise_response(value).reasoning == "非流式思考"
    assert _normalise_response({"content": "答案", "reasoning": "映射思考"}).reasoning == "映射思考"


def test_client_reports_non_streamed_thinking_before_the_answer(monkeypatch):
    monkeypatch.setenv("BATTERYSCIENTIST_STREAM", "false")
    message = SimpleNamespace(content="答案", tool_calls=None, reasoning_content="思考过程")
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
    )
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion))
    )
    seen = []
    client = ModelClient("demo", client=transport)
    response = client.complete(
        [{"role": "user", "content": "问"}],
        [],
        on_delta=lambda text: seen.append(("answer", text)),
        on_reasoning=lambda text: seen.append(("thinking", text)),
    )
    assert response.reasoning == "思考过程"
    assert seen == [("thinking", "思考过程"), ("answer", "答案")]


def test_worker_merges_reasoning_deltas_separately_from_answer_deltas(monkeypatch):
    from batteryscientist.api import worker

    stored = []
    monkeypatch.setattr(worker, "event", lambda project, kind, data, run: stored.append((kind, data["content"])))
    with worker.buffered_events("p", "r") as emit:
        base = {"phase": "literature", "iteration": 1}
        emit({**base, "kind": "assistant_reasoning_delta", "content": "先"})
        emit({**base, "kind": "assistant_reasoning_delta", "content": "想"})
        emit({**base, "kind": "assistant_delta", "content": "答"})
        emit({**base, "kind": "assistant_delta", "content": "案"})
    assert stored == [("assistant_reasoning_delta", "先想"), ("assistant_delta", "答案")]


def test_token_counting_falls_back_when_the_tokenizer_cannot_download(monkeypatch):
    from batteryscientist.agent import context

    def offline(_name):
        raise OSError("Tunnel connection failed: 403 Forbidden")

    context._encoding.cache_clear()
    context.text_tokens.cache_clear()
    monkeypatch.setattr(context.tiktoken, "get_encoding", offline)
    try:
        assert context.text_tokens("跨电芯泛化") == context.estimated_tokens("跨电芯泛化") == 6
        assert context.text_tokens("cross cell") >= len("cross cell") // 4
    finally:
        context._encoding.cache_clear()
        context.text_tokens.cache_clear()
