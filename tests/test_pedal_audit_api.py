"""排练踏板日志校对接口的端到端测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _create_plan(client) -> str:
    # 分页结果：页 1 = 小节 0..1，页 2 = 小节 2..3
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [3, 3, 4, 2],
            "silence_seconds": [2, 5, 1, 0],
            "page_height": 6,
            "min_turn_time": 2,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _audit(client, plan_id, **overrides):
    payload = {
        "beats_per_measure": [4, 4, 4, 4],
        "tempo_bpm": 120,
        "repeats": [{"start_measure": 1, "end_measure": 2, "play_count": 2}],
        "events": [],
        "tolerance_seconds": 0.5,
    }
    payload.update(overrides)
    return client.post(
        f"/api/v1/plans/{plan_id}/rehearsal/pedal-audit", json=payload
    )


# 该方案 + 排练参数下的应翻页提示：
# 演奏序列 0,1,2,1(遍2),2(遍2),3；跨页边界 4s(->2)、6s(->1 回翻)、8s(->2)


def test_pedal_audit_happy_path(client):
    plan_id = _create_plan(client)
    resp = _audit(
        client,
        plan_id,
        events=[
            {"seconds": 4.1, "page_after": 2},
            {"seconds": 6.0, "page_after": 1},
            {"seconds": 7.9, "page_after": 2},
            {"seconds": 11.0, "page_after": 1},
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == plan_id
    assert body["total_duration"] == pytest.approx(12.0)

    assert body["cue_results"] == [
        {
            "cue_index": 0,
            "boundary_seconds": 4.0,
            "to_page": 2,
            "matched": True,
            "event_index": 0,
            "event_seconds": 4.1,
            "time_diff_seconds": 0.1,
        },
        {
            "cue_index": 1,
            "boundary_seconds": 6.0,
            "to_page": 1,
            "matched": True,
            "event_index": 1,
            "event_seconds": 6.0,
            "time_diff_seconds": 0.0,
        },
        {
            "cue_index": 2,
            "boundary_seconds": 8.0,
            "to_page": 2,
            "matched": True,
            "event_index": 2,
            "event_seconds": 7.9,
            "time_diff_seconds": -0.1,
        },
    ]
    # 第 4 个事件（11 秒翻到页 1）没有对应提示（演奏结束不生成提示）-> 多翻
    assert [m["extra"] for m in body["events"]] == [
        False, False, False, True
    ]
    assert body["events"][3] == {
        "event_index": 3,
        "seconds": 11.0,
        "page_after": 1,
        "extra": True,
        "cue_index": None,
    }
    assert body["missed_count"] == 0
    assert body["extra_count"] == 1
    assert body["error_count"] == 1
    assert body["total_time_diff_seconds"] == pytest.approx(0.2)


def test_pedal_audit_backward_cue_requires_target_page_one(client):
    plan_id = _create_plan(client)
    # 6 秒回翻提示要求操作后页码为 1；翻到页 2 不允许匹配 -> 该提示漏翻
    resp = _audit(
        client,
        plan_id,
        events=[
            {"seconds": 4.0, "page_after": 2},
            {"seconds": 6.0, "page_after": 2},
            {"seconds": 8.0, "page_after": 2},
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    matched = [(c["cue_index"], c["event_index"]) for c in body["cue_results"]
               if c["matched"]]
    assert matched == [(0, 0), (2, 2)]
    middle = body["cue_results"][1]
    assert middle["matched"] is False
    assert middle["event_index"] is None
    assert middle["to_page"] == 1
    assert body["missed_count"] == 1
    assert body["extra_count"] == 1


def test_pedal_audit_tolerance_exceeded_is_miss_and_extra(client):
    plan_id = _create_plan(client)
    # 4.6 秒距边界 4.0 已 0.6 秒，超出 0.5 容差 -> 3 条提示全漏翻、事件多翻
    resp = _audit(
        client, plan_id, events=[{"seconds": 4.6, "page_after": 2}]
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cue_results"][0]["matched"] is False
    assert body["events"][0]["extra"] is True
    assert body["missed_count"] == 3
    assert body["extra_count"] == 1
    assert body["error_count"] == 4
    assert body["total_time_diff_seconds"] == 0.0


def test_pedal_audit_empty_events_all_missed(client):
    plan_id = _create_plan(client)
    resp = _audit(client, plan_id, events=[])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["missed_count"] == 3
    assert body["extra_count"] == 0
    assert body["error_count"] == 3
    assert all(not c["matched"] for c in body["cue_results"])
    assert body["events"] == []


def test_pedal_audit_single_page_plan_has_no_cues(client):
    # 单页方案：无跨页提示，全部事件都是多翻
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1, 1],
            "silence_seconds": [1, 0],
            "page_height": 10,
            "min_turn_time": 1,
        },
    )
    plan_id = resp.json()["id"]
    resp = _audit(
        client,
        plan_id,
        beats_per_measure=[4, 4],
        repeats=[],
        events=[{"seconds": 1.0, "page_after": 1}],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cue_results"] == []
    assert body["missed_count"] == 0
    assert body["extra_count"] == 1


def test_pedal_audit_plan_not_found(client):
    resp = _audit(client, "does-not-exist")
    assert resp.status_code == 404
    assert "规划结果不存在" in resp.json()["detail"]


def test_pedal_audit_beats_length_mismatch(client):
    plan_id = _create_plan(client)
    resp = _audit(client, plan_id, beats_per_measure=[4, 4, 4])
    assert resp.status_code == 422
    assert "数组长度" in resp.json()["detail"]


def test_pedal_audit_overlapping_repeats(client):
    plan_id = _create_plan(client)
    resp = _audit(
        client,
        plan_id,
        repeats=[
            {"start_measure": 0, "end_measure": 2, "play_count": 2},
            {"start_measure": 1, "end_measure": 3, "play_count": 2},
        ],
    )
    assert resp.status_code == 422
    assert "重叠或嵌套" in resp.json()["detail"]


def test_pedal_audit_events_must_be_nondecreasing(client):
    plan_id = _create_plan(client)
    resp = _audit(
        client,
        plan_id,
        events=[
            {"seconds": 5.0, "page_after": 2},
            {"seconds": 4.0, "page_after": 1},
        ],
    )
    assert resp.status_code == 422
    assert "非降序" in resp.json()["detail"]


def test_pedal_audit_seconds_beyond_total_duration(client):
    plan_id = _create_plan(client)
    # 总时长 12 秒；12.0 端点允许，12.1 拒绝
    ok = _audit(client, plan_id, events=[{"seconds": 12.0, "page_after": 1}])
    assert ok.status_code == 200, ok.text
    bad = _audit(client, plan_id, events=[{"seconds": 12.1, "page_after": 1}])
    assert bad.status_code == 422
    assert "超出演奏总时长" in bad.json()["detail"]


def test_pedal_audit_negative_seconds(client):
    plan_id = _create_plan(client)
    resp = _audit(client, plan_id, events=[{"seconds": -0.1, "page_after": 1}])
    assert resp.status_code == 422


def test_pedal_audit_page_out_of_range(client):
    plan_id = _create_plan(client)
    resp = _audit(client, plan_id, events=[{"seconds": 4.0, "page_after": 3}])
    assert resp.status_code == 422
    assert "页码" in resp.json()["detail"]
    resp0 = _audit(client, plan_id, events=[{"seconds": 4.0, "page_after": 0}])
    assert resp0.status_code == 422


def test_pedal_audit_non_integer_page(client):
    plan_id = _create_plan(client)
    resp = _audit(
        client, plan_id, events=[{"seconds": 4.0, "page_after": 2.0}]
    )
    assert resp.status_code == 422
    resp_bool = _audit(
        client, plan_id, events=[{"seconds": 4.0, "page_after": True}]
    )
    assert resp_bool.status_code == 422


def test_pedal_audit_negative_tolerance(client):
    plan_id = _create_plan(client)
    resp = _audit(client, plan_id, tolerance_seconds=-0.01)
    assert resp.status_code == 422


def test_pedal_audit_missing_fields_and_extra_fields(client):
    plan_id = _create_plan(client)
    base = {
        "beats_per_measure": [4, 4, 4, 4],
        "tempo_bpm": 120,
        "events": [],
        "tolerance_seconds": 0.5,
    }
    for drop in ("events", "tolerance_seconds", "tempo_bpm"):
        bad = {k: v for k, v in base.items() if k != drop}
        r = client.post(
            f"/api/v1/plans/{plan_id}/rehearsal/pedal-audit", json=bad
        )
        assert r.status_code == 422, drop
    r = client.post(
        f"/api/v1/plans/{plan_id}/rehearsal/pedal-audit",
        json=dict(base, unexpected=1),
    )
    assert r.status_code == 422


def test_pedal_audit_does_not_modify_saved_plan(client):
    plan_id = _create_plan(client)
    before = client.get(f"/api/v1/plans/{plan_id}").json()
    assert _audit(client, plan_id).status_code == 200
    after = client.get(f"/api/v1/plans/{plan_id}").json()
    assert before == after


def test_pedal_audit_request_echoed(client):
    plan_id = _create_plan(client)
    resp = _audit(
        client,
        plan_id,
        events=[{"seconds": 4.0, "page_after": 2}],
        tolerance_seconds=1.0,
    )
    body = resp.json()
    assert body["request"]["events"] == [
        {"seconds": 4.0, "page_after": 2}
    ]
    assert body["request"]["tolerance_seconds"] == 1.0
    assert body["request"]["repeats"] == [
        {"start_measure": 1, "end_measure": 2, "play_count": 2}
    ]
