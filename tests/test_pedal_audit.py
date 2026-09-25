"""排练踏板日志校对核心的单元测试。运行：python -m pytest -q"""

from __future__ import annotations

import pytest

from app.pedal_audit import (
    MAX_EVENTS,
    CueAudit,
    PedalAuditError,
    PedalEvent,
    audit_pedal_log,
)


def _events(pairs):
    """pairs: [(秒数, 操作后页码), ...]"""
    return [PedalEvent(seconds=t, page_after=p) for t, p in pairs]


def _audit(cue_tp, event_tp, tol, duration=100.0, page_count=5, from_pages=None):
    """cue_tp/event_tp: [(秒数, 目标页), ...]"""
    cue_seconds = [t for t, _ in cue_tp]
    cue_to = [p for _, p in cue_tp]
    cue_from = from_pages or [max(1, p - 1) for p in cue_to]
    return audit_pedal_log(
        cue_seconds=cue_seconds,
        cue_to_pages=cue_to,
        cue_from_pages=cue_from,
        events=_events(event_tp),
        total_duration=duration,
        page_count=page_count,
        tolerance_seconds=tol,
    )


def _matched_indices(result):
    return tuple(
        a.pedal_event_index for a in result.cue_audits if a.matched
    )


def test_perfect_matching():
    r = _audit([(4.0, 2), (6.0, 1), (8.0, 2)], [(4.1, 2), (6.0, 1), (8.2, 2)], 0.5)
    assert r.total_error_count == 0
    assert r.missed_count == 0
    assert r.extra_count == 0
    assert r.extra_turns == []
    assert _matched_indices(r) == (0, 1, 2)
    assert r.total_time_diff_seconds == pytest.approx(0.3)
    assert all(a.matched for a in r.cue_audits)
    # 回显提示时刻与目标页
    assert [a.expected_seconds for a in r.cue_audits] == [4.0, 6.0, 8.0]
    assert [a.to_page for a in r.cue_audits] == [2, 1, 2]


def test_missed_and_extra_counts():
    # 三条提示，只踩了一次且页码不符；另有一次多翻
    r = _audit([(4.0, 2), (6.0, 1), (8.0, 2)], [(4.0, 3), (9.0, 2)], 0.5)
    # 事件 0 页码 3 与任何提示目标页都不同（提示目标为 2/1/2），不能匹配；
    # 事件 1 (9.0→2) 与提示 2 (8.0→2) 差 1.0 > 0.5，也不能匹配
    assert r.total_error_count == 5  # 3 漏翻 + 2 多翻
    assert r.missed_count == 3
    assert r.extra_count == 2
    assert r.extra_turns == [0, 1]
    assert all(not a.matched for a in r.cue_audits)
    assert r.total_time_diff_seconds == 0.0


def test_page_must_match_even_if_time_close():
    # 时间恰好一致但目标页不同，不得匹配
    r = _audit([(4.0, 2)], [(4.0, 1)], 1.0)
    assert r.total_error_count == 2
    assert not r.cue_audits[0].matched


def test_tolerance_boundary_inclusive():
    # 时间差恰好等于容差允许匹配
    r = _audit([(4.0, 2)], [(4.5, 2)], 0.5)
    assert r.total_error_count == 0
    assert r.cue_audits[0].matched
    assert r.cue_audits[0].time_diff_seconds == pytest.approx(0.5)
    # 略超容差不匹配
    r2 = _audit([(4.0, 2)], [(4.5 + 1e-6, 2)], 0.5)
    assert r2.total_error_count == 2


def test_one_to_one_global_alignment():
    # 两条目标页相同的提示，三个候选事件；只能一对一
    # 提示 10、20；事件 10、11、20，容差 1
    r = _audit([(10.0, 2), (20.0, 2)], [(10.0, 2), (11.0, 2), (20.0, 2)], 1.0)
    assert r.total_error_count == 1  # 两次匹配 + 一次多翻
    assert _matched_indices(r) == (0, 2)
    assert r.extra_turns == [1]


def test_primary_goal_maximizes_matches_over_time_sum():
    # 选事件 j 给前一条会阻断后一条匹配：全局应优先匹配数最多
    # 提示 10、10.6；事件 10、10.6（页均为 2），容差 0.5
    # 若贪心就近：提示10 可选 10 或 10.5? 构造迫使错配阻断
    r = _audit(
        [(10.0, 2), (10.6, 2)],
        [(10.0, 2), (10.5, 2)],
        0.5,
    )
    # 两种对齐都恰好 2 次匹配：(10->10,10.6->10.5 差0.1) 与 (10->10.5 差0.5,10.6? 无)
    assert r.total_error_count == 0
    assert _matched_indices(r) == (0, 1)
    assert r.total_time_diff_seconds == pytest.approx(0.1)


def test_secondary_goal_minimizes_time_diff_sum():
    # 同样匹配数下取时间差之和更小的对齐
    # 提示 10、20；事件 9.8 与 10.2 都能匹配提示 10，20 无事件
    # 匹配数均为 1；选差 0.2（事件 1）而非 0... 实际 9.8 差0.2,10.2 差0.2 并列 ->
    # 改为明显不同
    r = _audit([(10.0, 2), (20.0, 2)], [(9.0, 2), (9.9, 2)], 1.5)
    # 两条事件都只在提示10容差内；最多匹配 1（取差 0.1 的事件 1），
    # 提示20 漏翻、事件0 多翻，共 2 项
    assert r.total_error_count == 2
    assert _matched_indices(r) == (1,)
    assert r.total_time_diff_seconds == pytest.approx(0.1)
    assert r.extra_turns == [0]


def test_tertiary_goal_lexicographic_min_event_indices():
    # 漏翻+多翻与时间差之和都相同，取匹配事件序号序列字典序最小
    # 提示 10、20；事件 10、20 与另一组等价选择：用两条提示都可匹配多个等距事件
    # 提示 10、20；事件 10、10（同刻两事件）、20，容差 0
    # 提示10 可匹配事件0或1（时间差均0），字典序取事件0
    r = _audit(
        [(10.0, 2), (20.0, 2)],
        [(10.0, 2), (10.0, 2), (20.0, 2)],
        0.0,
    )
    assert r.total_error_count == 1
    assert _matched_indices(r) == (0, 2)
    assert r.extra_turns == [1]


def test_lexicographic_when_first_index_tied():
    # 首个匹配序号相同、第二个序号并列时取更小
    # 来自暴力对撞用例：匹配数与时间差和相同
    cues = [
        (0.0, 3), (3.0, 3), (3.5, 1), (5.0, 1), (8.0, 2), (18.0, 3)
    ]
    events = [
        (4.5, 1), (4.5, 3), (5.5, 2), (6.5, 3), (7.0, 1), (8.0, 3),
        (9.0, 1), (12.5, 1),
    ]
    r = _audit(cues, events, 3.0)
    # 与全元组暴力 DP 一致：匹配事件序列 (0, 2)
    assert _matched_indices(r) == (0, 2)
    assert r.total_time_diff_seconds == pytest.approx(3.0)


def test_empty_events_all_cues_missed():
    r = _audit([(4.0, 2), (6.0, 1)], [], 1.0)
    assert r.total_error_count == 2
    assert r.missed_count == 2
    assert r.extra_count == 0
    assert all(not a.matched for a in r.cue_audits)


def test_empty_cues_all_events_extra():
    r = _audit([], [(1.0, 2), (2.0, 3)], 1.0, page_count=5)
    assert r.cue_audits == []
    assert r.extra_turns == [0, 1]
    assert r.extra_count == 2
    assert r.total_error_count == 2
    assert r.total_time_diff_seconds == 0.0


def test_events_non_descending_equal_times_allowed():
    # 时间非降序：相等允许
    r = _audit(
        [(10.0, 2)], [(5.0, 2), (5.0, 2)], 10.0
    )
    # 两条同刻事件只有一条能匹配；字典序取事件 0
    assert _matched_indices(r) == (0,)
    assert r.extra_turns == [1]


def test_event_at_total_duration_rejected():
    with pytest.raises(PedalAuditError, match="越出演奏时长"):
        _audit([], [(12.0, 1)], 0.0, duration=12.0)


def test_event_beyond_total_duration_rejected():
    with pytest.raises(PedalAuditError, match="越出演奏时长"):
        _audit([], [(12.5, 1)], 0.0, duration=12.0)


def test_negative_seconds_rejected():
    with pytest.raises(PedalAuditError, match="不能为负数"):
        _audit([], [(-0.1, 1)], 0.0, duration=12.0)


def test_page_out_of_range_rejected():
    with pytest.raises(PedalAuditError, match="越出页码范围"):
        _audit([], [(1.0, 6)], 0.0, page_count=5)
    with pytest.raises(PedalAuditError, match="越出页码范围"):
        _audit([], [(1.0, 0)], 0.0, page_count=5)


def test_descending_events_rejected():
    with pytest.raises(PedalAuditError, match="非降序"):
        audit_pedal_log(
            cue_seconds=[1.0],
            cue_to_pages=[2],
            cue_from_pages=[1],
            events=_events([(5.0, 2), (4.0, 2)]),
            total_duration=10.0,
            page_count=5,
            tolerance_seconds=0.0,
        )


def test_negative_and_non_finite_tolerance_rejected():
    with pytest.raises(PedalAuditError, match="不能为负数"):
        _audit([], [], -0.1)
    with pytest.raises(PedalAuditError, match="有限数值"):
        _audit([], [], float("inf"))
    with pytest.raises(PedalAuditError, match="有限数值"):
        _audit([], [], float("nan"))


def test_non_finite_and_bool_event_fields_rejected():
    with pytest.raises(PedalAuditError, match="有限数值"):
        _audit([], [(float("nan"), 1)], 0.0)
    with pytest.raises(PedalAuditError, match="整数"):
        audit_pedal_log(
            cue_seconds=[1.0],
            cue_to_pages=[2],
            cue_from_pages=[1],
            events=[PedalEvent(1.0, True)],  # type: ignore[arg-type]
            total_duration=10.0,
            page_count=5,
            tolerance_seconds=0.0,
        )


def test_too_many_events_rejected():
    events = _events([(float(i % 50), 1) for i in range(MAX_EVENTS + 1)])
    with pytest.raises(PedalAuditError, match=f"超过上限 {MAX_EVENTS}"):
        audit_pedal_log(
            cue_seconds=[],
            cue_to_pages=[],
            cue_from_pages=[],
            events=events,
            total_duration=10.0 ** 9,
            page_count=5,
            tolerance_seconds=0.0,
        )


def test_result_is_deterministic():
    cues = [(4.0, 2), (6.0, 1), (8.0, 2)]
    events = [(4.1, 2), (6.0, 1), (8.2, 2)]
    r1 = _audit(cues, events, 0.5)
    r2 = _audit(cues, events, 0.5)
    assert _matched_indices(r1) == _matched_indices(r2)
    assert r1.total_time_diff_seconds == r2.total_time_diff_seconds


def test_backward_cue_target_page_drives_match():
    # 重复段回跳造成的回翻提示目标页较小（如 to_page=1）；
    # 只有操作后页码=1 的事件能匹配
    r = _audit(
        [(4.0, 2), (6.0, 1), (8.0, 2)],
        [(6.05, 1)],
        0.5,
        from_pages=[1, 2, 1],
    )
    # 仅中间回翻提示匹配唯一一条事件（事件序号 0），其余漏翻
    assert _matched_indices(r) == (0,)
    mid = r.cue_audits[1]
    assert mid.from_page == 2 and mid.to_page == 1
    assert mid.pedal_event_index == 0
    assert mid.pedal_seconds == pytest.approx(6.05)


# ---------------------------------------------------------------- 随机暴力对照


def _brute_force(cues, events, tol):
    """全元组后缀 DP：(漏翻+多翻, 时间差和, 匹配事件序号元组) 整体字典序。"""
    m, n = len(cues), len(events)
    dp = [[None] * (n + 1) for _ in range(m + 1)]
    dp[m][n] = (0, 0.0, ())
    for i in range(m):
        dp[i][n] = (m - i, 0.0, ())
    for j in range(n + 1):
        dp[m][j] = (n - j, 0.0, ())
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            candidates = [
                (dp[i + 1][j][0] + 1, dp[i + 1][j][1], dp[i + 1][j][2]),
                (dp[i][j + 1][0] + 1, dp[i][j + 1][1], dp[i][j + 1][2]),
            ]
            ct, cp = cues[i]
            et, ep = events[j]
            if cp == ep and abs(ct - et) <= tol + 1e-9 * max(1.0, tol):
                g, s, t = dp[i + 1][j + 1]
                candidates.append((g, s + abs(ct - et), (j,) + t))
            dp[i][j] = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    return dp[0][0]


@pytest.mark.parametrize("seed", range(300))
def test_random_alignment_matches_brute_force(seed):
    import random

    rng = random.Random(seed)
    m = rng.randint(0, 8)
    n = rng.randint(0, 8)
    page_count = rng.randint(1, 4)
    cues = sorted(
        (round(rng.uniform(0, 40), 2), rng.randint(1, page_count)) for _ in range(m)
    )
    events = sorted(
        (round(rng.uniform(0, 40), 2), rng.randint(1, page_count)) for _ in range(n)
    )
    tol = rng.choice([0.0, 0.5, 1.0, 3.0, 40.0])

    r = _audit(cues, events, tol, duration=1000.0, page_count=page_count)
    expected = _brute_force(cues, events, tol)

    assert r.total_error_count == expected[0]
    assert r.total_time_diff_seconds == expected[1]
    assert _matched_indices(r) == expected[2]
    # 不变量：匹配数 + 漏翻数 = 提示数；匹配数 + 多翻数 = 事件数
    matched = sum(1 for a in r.cue_audits if a.matched)
    assert matched + r.missed_count == m
    assert matched + r.extra_count == n
    assert sorted(a.pedal_event_index for a in r.cue_audits if a.matched) == sorted(
        set(range(n)) - set(r.extra_turns)
    )


def test_large_scale_performance():
    """500 提示 × 500 事件的最坏情形应在合理时间内完成。"""
    import time

    cues = [(float(i), 2) for i in range(500)]
    events = [(float(i) + 0.01, 2) for i in range(500)]
    start = time.time()
    r = _audit(cues, events, 0.5, duration=100000.0, page_count=5)
    elapsed = time.time() - start
    assert r.total_error_count == 0
    assert _matched_indices(r) == tuple(range(500))
    assert elapsed < 10.0

