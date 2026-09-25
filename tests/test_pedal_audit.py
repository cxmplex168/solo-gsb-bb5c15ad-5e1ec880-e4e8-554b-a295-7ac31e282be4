"""排练踏板日志校对核心的单元测试。运行：python -m pytest -q"""

from __future__ import annotations

import random

import pytest

from app.pedal_audit import (
    PedalAuditError,
    PedalEvent,
    audit_pedal_log,
)


def _audit(bounds, pages, events, total=12.0, page_count=2, tol=0.5):
    return audit_pedal_log(
        cue_boundaries=bounds,
        cue_target_pages=pages,
        events=events,
        total_duration=total,
        page_count=page_count,
        tolerance_seconds=tol,
    )


def _pairs(result):
    return [
        (c.cue_index, c.event_index)
        for c in result.cue_results
        if c.matched
    ]


# --------------------------------------------------------------- 基本对齐


def test_perfect_matching():
    r = _audit(
        [4.0, 6.0, 8.0],
        [2, 1, 2],
        [(4.1, 2), (6.0, 1), (7.9, 2)],
    )
    assert r.error_count == 0
    assert r.missed_count == 0
    assert r.extra_count == 0
    assert r.total_time_diff == pytest.approx(0.2)
    assert _pairs(r) == [(0, 0), (1, 1), (2, 2)]
    assert [c.time_diff_seconds for c in r.cue_results] == [0.1, 0.0, -0.1]
    assert all(not m.extra for m in r.event_marks)


def test_all_missed_and_all_extra():
    # 页码不同：提示全漏翻、事件全多翻
    r = _audit([4.0, 8.0], [1, 1], [(4.0, 2), (8.0, 2)])
    assert r.missed_count == 2
    assert r.extra_count == 2
    assert r.error_count == 4
    assert r.total_time_diff == 0.0
    assert all(not c.matched for c in r.cue_results)
    assert all(m.extra for m in r.event_marks)


def test_empty_cues_all_events_extra():
    r = _audit([], [], [(1.0, 1), (2.0, 2)], page_count=2)
    assert r.cue_results == []
    assert r.extra_count == 2
    assert r.error_count == 2


def test_empty_events_all_cues_missed():
    r = _audit([4.0, 6.0, 8.0], [2, 1, 2], [])
    assert r.event_marks == []
    assert r.missed_count == 3
    assert r.error_count == 3


def test_empty_inputs():
    r = _audit([], [], [])
    assert r.error_count == 0
    assert r.total_time_diff == 0.0


def test_matching_requires_same_target_page():
    # 时间完全一致但目标页不同 -> 不可匹配
    r = _audit([2.0], [2], [(2.0, 1)], page_count=3)
    assert r.error_count == 2
    r2 = _audit([2.0], [2], [(2.0, 2)], page_count=3, tol=0.0)
    assert r2.error_count == 0


def test_tolerance_boundary_is_inclusive():
    # 时间差恰好等于容差允许匹配
    r = _audit([4.0], [1], [(4.5, 1)], tol=0.5)
    assert r.error_count == 0
    r2 = _audit([4.0], [1], [(4.5000001, 1)], tol=0.5)
    assert r2.error_count == 2


def test_events_at_zero_and_total_duration_allowed():
    r = _audit([0.0], [1], [(0.0, 1)], total=10.0, tol=0.0)
    assert r.error_count == 0
    r2 = _audit([10.0], [1], [(10.0, 1)], total=10.0, tol=0.0)
    assert r2.error_count == 0


def test_equal_timestamps_nondecreasing_allowed():
    # 同一时刻两个事件分别配同一时刻的两条提示（页码不同）
    r = _audit(
        [2.0, 2.0], [1, 2], [(2.0, 1), (2.0, 2)], page_count=2, tol=0.0
    )
    assert r.error_count == 0
    assert _pairs(r) == [(0, 0), (1, 1)]


def test_one_to_one_no_event_reuse():
    # 只有一个事件落在两条提示的容差窗内：只能配一条，另一条必漏翻
    r = _audit([2.0, 2.4], [1, 1], [(2.2, 1)], tol=0.5)
    assert r.error_count == 1  # 1 匹配 + 1 漏翻
    assert r.missed_count == 1
    assert r.extra_count == 0
    # 时间差更小（0.2 vs 0.4）的第 0 条提示赢得该事件
    assert _pairs(r) == [(0, 0)]


# --------------------------------------------------------------- 优化目标并列


def test_prefers_more_matches_over_time_diff():
    # 一配一会产生多翻/漏翻；两配两虽时间差更大，但错误总数更少 -> 取两配两
    r = _audit(
        [2.0, 4.0], [1, 1], [(2.4, 1), (4.4, 1)], tol=0.5
    )
    assert r.error_count == 0
    assert _pairs(r) == [(0, 0), (1, 1)]
    assert r.total_time_diff == pytest.approx(0.8)


def test_tie_on_errors_prefers_smaller_time_diff_sum():
    # 两种对齐都只有 0 错误时（这里构造错误数相同、时间差不同的一对一对齐）
    r = _audit([2.0, 4.0], [1, 1], [(2.0, 1), (4.0, 1)], tol=5.0)
    assert _pairs(r) == [(0, 0), (1, 1)]
    assert r.total_time_diff == pytest.approx(0.0)

    # 交叉配（0-1,1-0）时间差更大，最优必须是顺序配
    r2 = _audit([2.0, 3.0], [1, 1], [(2.0, 1), (3.0, 1)], tol=5.0)
    assert _pairs(r2) == [(0, 0), (1, 1)]


def test_tie_on_both_prefers_lex_smallest_event_sequence():
    # 两条提示都能配两个事件，且两种完美配对的时间差之和相同：
    # bounds 2.0/4.0，events 2.0/4.0 -> (0,1) 序列 vs 不可能更优的交叉
    # 构造对称情形：bounds 2/4，events 2/4；seq 必为 (0,1)
    r = _audit([2.0, 4.0], [1, 1], [(2.0, 1), (4.0, 1)], tol=5.0)
    assert tuple(c.event_index for c in r.cue_results) == (0, 1)

    # 第一条提示可在事件 0、1 间任选且总时间差相同：取较小事件序号
    # bounds [2.0, 6.0]，events [2.0, 2.0, 6.0]，容差 5：
    # 配法 A：(0,2) 时间差 |2-2|+|6-6|=0；配法 B：(1,2) 同样 0 -> 取 A
    r2 = _audit(
        [2.0, 6.0], [1, 1], [(2.0, 1), (2.0, 1), (6.0, 1)], tol=5.0
    )
    assert tuple(c.event_index for c in r2.cue_results) == (0, 2)
    assert r2.event_marks[1].extra is True


def test_same_event_sequence_tie_prefers_earlier_cue():
    # 事件子集相同但配到不同提示：同样的事件分别能配 cue0/cue1，
    # 序列相同（都只配一个事件 0）时，优先较早提示
    r = _audit([2.0, 2.5], [1, 1], [(2.25, 1)], tol=0.5)
    # 配 cue0 时差 0.25、配 cue1 时差 0.25，完全并列 -> cue0
    assert _pairs(r) == [(0, 0)]
    assert r.cue_results[1].matched is False


def test_order_preserving_alignment():
    # 事件顺序不能颠倒匹配：bounds 2/4，events 2/4 大容差下禁止 (1,0)
    r = _audit([2.0, 4.0], [1, 1], [(2.0, 1), (4.0, 1)], tol=10.0)
    assert _pairs(r) == [(0, 0), (1, 1)]


# --------------------------------------------------------------- 入参校验


def test_events_must_be_nondecreasing():
    with pytest.raises(PedalAuditError, match="非降序"):
        _audit([2.0], [1], [(3.0, 1), (1.0, 1)])


def test_negative_seconds_rejected():
    with pytest.raises(PedalAuditError, match="负数"):
        _audit([2.0], [1], [(-0.001, 1)])


def test_seconds_beyond_total_rejected():
    with pytest.raises(PedalAuditError, match="超出演奏总时长"):
        _audit([2.0], [1], [(12.0001, 1)], total=12.0)


def test_page_out_of_range_rejected():
    with pytest.raises(PedalAuditError, match="页码"):
        _audit([2.0], [1], [(2.0, 0)])
    with pytest.raises(PedalAuditError, match="页码"):
        _audit([2.0], [1], [(2.0, 3)], page_count=2)


def test_non_integer_page_rejected():
    with pytest.raises(PedalAuditError, match="整数"):
        _audit([2.0], [1], [(2.0, 1.5)])  # type: ignore[list-item]
    with pytest.raises(PedalAuditError, match="整数"):
        _audit([2.0], [1], [(2.0, True)])  # type: ignore[list-item]


def test_non_finite_seconds_rejected():
    with pytest.raises(PedalAuditError, match="有限数值"):
        _audit([2.0], [1], [(float("nan"), 1)])
    with pytest.raises(PedalAuditError, match="有限数值"):
        _audit([2.0], [1], [(float("inf"), 1)])


def test_negative_tolerance_rejected():
    with pytest.raises(PedalAuditError, match="时间容差"):
        _audit([2.0], [1], [(2.0, 1)], tol=-0.1)


def test_too_many_events_rejected():
    events = [(1.0, 1)] * 501
    with pytest.raises(PedalAuditError, match="上限"):
        _audit([1.0], [1], events, total=1000.0)


def test_pedal_event_dataclass_input_accepted():
    r = audit_pedal_log(
        [2.0],
        [1],
        [PedalEvent(2.0, 1)],
        total_duration=4.0,
        page_count=2,
        tolerance_seconds=0.0,
    )
    assert r.error_count == 0


# --------------------------------------------------------------- 暴力对照


def _brute_force(bounds, pages, events, tol):
    """枚举全部保序匹配的参考实现：返回 (错误数, 时间差和, 事件序号, 提示序号)。"""
    n, m = len(bounds), len(events)
    best = None

    def rec(i, j, pairs):
        nonlocal best
        if i == n:
            matched = len(pairs)
            err = n + m - 2 * matched
            diff = sum(abs(events[k][0] - bounds[q]) for q, k in pairs)
            key = (
                err,
                diff,
                tuple(k for _, k in pairs),
                tuple(q for q, _ in pairs),
            )
            if best is None or key < best:
                best = key
            return
        rec(i + 1, j, pairs)  # 漏翻提示 i
        for k in range(j, m):
            if (
                events[k][1] == pages[i]
                and abs(events[k][0] - bounds[i]) <= tol + 1e-12
            ):
                rec(i + 1, k + 1, pairs + [(i, k)])

    rec(0, 0, [])
    return best


def test_random_cases_match_brute_force():
    rng = random.Random(20260925)
    for _ in range(2000):
        n = rng.randint(0, 7)
        m = rng.randint(0, 7)
        page_count = rng.randint(1, 4)
        bounds = sorted(round(rng.uniform(0, 9.5), 3) for _ in range(n))
        pages = [rng.randint(1, page_count) for _ in range(n)]
        times = sorted(round(rng.uniform(0, 10), 3) for _ in range(m))
        events = [(t, rng.randint(1, page_count)) for t in times]
        tol = rng.choice([0.0, 0.001, 0.3, 1.0, 2.5, 8.0])

        result = _audit(bounds, pages, events, total=10.0,
                        page_count=page_count, tol=tol)
        err, diff, seq, cues = _brute_force(bounds, pages, events, tol)

        got_seq = tuple(c.event_index for c in result.cue_results if c.matched)
        got_cues = tuple(c.cue_index for c in result.cue_results if c.matched)
        assert result.error_count == err
        assert result.total_time_diff == pytest.approx(diff, abs=1e-6)
        assert got_seq == seq
        assert got_cues == cues
        used = set(got_seq)
        assert sum(m.extra for m in result.event_marks) == m - len(used)


def test_performance_with_many_cues_and_events():
    rng = random.Random(99)
    bounds = sorted(rng.uniform(0, 1000) for _ in range(800))
    pages = [rng.randint(1, 5) for _ in bounds]
    times = sorted(rng.uniform(0, 1000) for _ in range(400))
    events = [(t, rng.randint(1, 5)) for t in times]
    import time

    start = time.perf_counter()
    result = _audit(bounds, pages, events, total=1000.0,
                    page_count=5, tol=5.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0
    # 结构守恒：匹配数 + 漏翻数 = 提示数；匹配数 + 多翻数 = 事件数
    matched = sum(c.matched for c in result.cue_results)
    assert matched + result.missed_count == len(bounds)
    assert matched + result.extra_count == len(events)
