"""Writing cards use original passages and leave scientific Wiki nodes intact."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from batteryscientist.agent.loop import tool_schema
from batteryscientist.tools import BUILTIN_TOOLS
from batteryscientist.tools.core import ToolContext
from batteryscientist.tools.profiles import tools_for_phase
from batteryscientist.tools.wiki.search import read_wiki
from batteryscientist.tools.wiki.writing import (
    extract_card,
    read_writing_card,
    search_writing_cards,
)

QUOTE = "The dominant approach requires a complete history of the target cell."


def card():
    return {
        "paper_type": "方法研究",
        "motivation_type": "观测条件缺口",
        "one_line_claim": "在明确的输入条件下预测未来轨迹。",
        "moves": [
            {
                "function": name,
                "summary": "从场景逐步引出机制。",
                "source_ids": ["p1-b1"],
            }
            for name in ["场景", "缺口", "方案"]
        ],
        "key_passages": [
            {
                "quote": QUOTE,
                "source_id": "p1-b1",
                "why_effective": "具体交代输入条件。",
            }
            for _ in range(2)
        ],
        "lessons": [
            {
                "principle": name,
                "explanation": "通过条件建立因果联系。",
                "source_ids": ["p1-b1"],
            }
            for name in ["解释缺口", "衔接方法"]
        ],
        "transferable_skeleton": [
            "[场景]需要[能力]。",
            "[旧方法]依赖[缺失输入]。",
            "[机制]利用[可用输入]。",
        ],
    }


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    monkeypatch.setenv("BATTERYSCIENTIST_WIKI_ROOT", str(root))
    paper = root / "literature/papers/example.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(
        "# Example battery paper\n\n## Abstract\nKnowledge remains intact.\n"
    )
    evidence = root / "literature/sources/doc.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "document_id": "doc",
                "pdf_sha256": "original",
                "blocks": [{"id": "p1-b1", "page": 1, "type": "text", "text": QUOTE}],
            }
        )
    )
    source = root / "literature/metadata/example/source.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "document_id": "doc",
                "pdf_sha256": "original",
                "evidence_path": "wiki/literature/sources/doc.json",
            }
        )
    )
    return root, paper, evidence, ToolContext(tmp_path)


def model(value=None, callback=None):
    calls = []

    def complete(messages):
        calls.append(messages)
        if callback:
            callback()
        return json.dumps(value if value is not None else card(), ensure_ascii=False)

    return SimpleNamespace(model="test", complete_text=complete, calls=calls)


def test_extract_cache_read_search_and_preserve_knowledge(library):
    root, paper, evidence, ctx = library
    originals = {
        p: p.read_bytes()
        for p in [paper, evidence, paper.parent.parent / "metadata/example/source.json"]
    }
    fake = model()
    result = extract_card(ctx, "paper:example", user_goal="解释输入缺口", model=fake)
    assert result["status"] == "created"
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["content"].find(QUOTE) >= 0
    assert all(p.read_bytes() == content for p, content in originals.items())
    cached = extract_card(
        ctx, "literature/papers/example.md", user_goal="解释输入缺口", model=fake
    )
    assert cached["status"] == "cached" and len(fake.calls) == 1
    read = json.loads(read_writing_card(ctx, result["markdown_path"]).output)
    assert not read["stale"]
    assert "PDF p." not in read["markdown"]
    assert "p1-b1" not in read["markdown"]
    assert "来源记录" not in read["markdown"]
    assert "适用条件与不可照搬之处" not in read["markdown"]
    assert read["record"]["source_blocks"][0]["source_id"] == "p1-b1"
    search = json.loads(search_writing_cards(ctx, "输入缺口").output)
    assert search["matches"][0]["paper_id"] == "paper:example"
    assert not search["matches"][0]["stale"]
    wiki = json.loads(read_wiki(ctx, "literature/papers/example.md").output)
    assert wiki["writing_cards"][0]["card_path"] == result["card_path"]
    record = json.loads((root / result["card_path"]).read_text())
    assert (
        record["markdown_sha256"]
        == hashlib.sha256((root / result["markdown_path"]).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("invalid", ["quote", "source_id", "source_ids"])
def test_fabricated_quote_or_locator_never_publishes(library, invalid):
    root, _, _, ctx = library
    value = card()
    if invalid == "quote":
        value["key_passages"][0][
            "quote"
        ] = "This fabricated quote does not occur anywhere in the original source document."
    elif invalid == "source_id":
        value["key_passages"][0]["source_id"] = "p999-b1"
    else:
        value["moves"][0]["source_ids"] = ["p999-b1"]
    with pytest.raises(ValueError):
        extract_card(ctx, "paper:example", model=model(value))
    assert not (root / "writing").exists()


def test_changed_source_invalidates_cache_and_detects_midflight_changes(library):
    root, paper, _, ctx = library
    first = extract_card(ctx, "paper:example", model=model())
    paper.write_text(paper.read_text() + "Changed knowledge.\n")
    assert json.loads(read_writing_card(ctx, first["card_path"]).output)["stale"]
    fake = model()
    assert extract_card(ctx, "paper:example", model=fake)["status"] == "created"
    assert len(fake.calls) == 1
    old = (root / first["card_path"]).read_bytes()
    with pytest.raises(ValueError, match="Source changed"):
        extract_card(
            ctx,
            "paper:example",
            force=True,
            model=model(
                callback=lambda: paper.write_text("# Changed during model call")
            ),
        )
    assert (root / first["card_path"]).read_bytes() == old


def test_user_goals_and_focus_have_separate_cards(library):
    _, _, _, ctx = library
    paths = {
        extract_card(ctx, "paper:example", focus=focus, user_goal=goal, model=model())[
            "card_path"
        ]
        for focus, goal in [
            ("introduction", "缺口"),
            ("introduction", "贡献"),
            ("abstract", "缺口"),
        ]
    }
    assert len(paths) == 3


@pytest.mark.parametrize(
    "paper_id", ["../outside.md", "/tmp/outside.md", "paper:../example"]
)
def test_path_escape_rejected(library, paper_id):
    _, _, _, ctx = library
    with pytest.raises(ValueError):
        extract_card(ctx, paper_id, model=model())
    with pytest.raises(ValueError):
        read_writing_card(ctx, paper_id)


def test_source_document_mismatch_rejected(library):
    _, _, evidence, ctx = library
    value = json.loads(evidence.read_text())
    value["pdf_sha256"] = "another-pdf"
    evidence.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="do not match"):
        extract_card(ctx, "paper:example", model=model())


def test_tools_available_to_wiki_and_readable_by_writer():
    wiki = {s.name for s in tools_for_phase("wiki", BUILTIN_TOOLS)}
    writer = {s.name for s in tools_for_phase("paper_write", BUILTIN_TOOLS)}
    assert {"extract_writing_card", "read_writing_card", "search_writing_cards"} <= wiki
    assert {"read_writing_card", "search_writing_cards"} <= writer
    assert "extract_writing_card" not in writer
    spec = next(s for s in BUILTIN_TOOLS if s.name == "extract_writing_card")
    assert tool_schema(spec)["function"]["parameters"]["properties"]["focus"][
        "enum"
    ] == ["introduction", "abstract", "method", "experiments", "full"]
