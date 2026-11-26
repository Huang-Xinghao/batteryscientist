"""Process scans behind status polling are cached; starting a run always rescans."""

import pytest

from batteryscientist.api import app as api


@pytest.fixture
def scans(monkeypatch):
    calls = []

    def fake_scan():
        calls.append(1)
        return frozenset({f"/projects/{len(calls)}"})

    clock = [100.0]
    monkeypatch.setattr(api, "scan_terminal_projects", fake_scan)
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api, "_terminal_cache", None)
    return calls, clock


def test_repeated_status_reads_reuse_one_scan(scans):
    calls, _ = scans
    first = api.terminal_projects()
    second = api.terminal_projects()
    assert first == second == frozenset({"/projects/1"})
    assert len(calls) == 1


def test_cache_expires_after_ttl(scans):
    calls, clock = scans
    api.terminal_projects()
    clock[0] += api.TERMINAL_SCAN_TTL + 0.1
    assert api.terminal_projects() == frozenset({"/projects/2"})
    assert len(calls) == 2


def test_fresh_scan_bypasses_cache_and_refreshes_it(scans):
    calls, _ = scans
    api.terminal_projects()
    assert api.terminal_projects(fresh=True) == frozenset({"/projects/2"})
    assert api.terminal_projects() == frozenset({"/projects/2"})
    assert len(calls) == 2
