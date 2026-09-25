"""planner 核心算法的单元测试。运行：python -m pytest -q"""

from __future__ import annotations

import random
from itertools import product

import pytest

from app.planner import PlanningError, plan_pages


def _ranges(plan):
    return [(p.start_measure, p.end_measure) for p in plan.pages]


def brute_force_best(heights, silences, page_height, min_turn_time):
    """枚举所有断页位置组合，按目标字典序排序取最优；无方案返回 None。"""
    n = len(heights)
    best = None

    for mask in product([0, 1], repeat=n - 1):
        breaks = [b + 1 for b, bit in enumerate(mask) if bit]
        bounds = [0] + breaks + [n]
        ok = True
        turn_values = []
        for idx in range(len(bounds) - 1):
            a, e = bounds[idx], bounds[idx + 1]
            if sum(heights[a:e]) > page_height + 1e-9:
                ok = False
                break
            if e < n and silences[e - 1] + 1e-9 < min_turn_time:
                ok = False
                break
            if e < n:
                turn_values.append(silences[e - 1])
        if not ok:
            continue
        key = (
            len(bounds) - 1,                            # 页数最少
            -min(turn_values) if turn_values else -float("inf"),  # 最短静默最大（无翻页点视为无穷大）
            breaks,                                      # 断页序号字典序最小
        )
        if best is None or key < best[0]:
            best = (key, breaks)
    return None if best is None else best[1]


def test_single_page():
    plan = plan_pages(
        heights=[3, 3, 3],
        silences=[1, 1, 0],
        page_height=10,
        min_turn_time=1,
    )
    assert plan.page_count == 1
    assert plan.breaks == []
    assert plan.shortest_turn_silence is None
    assert _ranges(plan) == [(0, 2)]


def test_min_pages_only_feasible_turn_positions():
    # 容量 2 的 5 小节；小节 3 后静默 0.1 不达标，只能在位置 1、3 之后翻页。
    plan = plan_pages(
        heights=[1] * 5,
        silences=[10, 0.1, 1, 100, 0],
        page_height=2,
        min_turn_time=1,
    )
    assert plan.page_count == 3
    assert plan.breaks == [1, 3]
    assert plan.shortest_turn_silence == 1
    assert _ranges(plan) == [(0, 0), (1, 2), (3, 4)]
    assert [p.turn_silence for p in plan.pages] == [10, 1, None]


def test_maximin_prefers_larger_min_silence():
    # 容量 3 的 5 小节，2 页切法：b=1(min=1), b=2(min=5), b=3(min=9)
    plan = plan_pages(
        heights=[1] * 5,
        silences=[1, 5, 9, 1, 0],
        page_height=3,
        min_turn_time=1,
    )
    assert plan.page_count == 2
    assert plan.breaks == [3]
    assert plan.shortest_turn_silence == 9


def test_lexicographically_smallest_breaks_on_tie():
    # 两个 2 页方案最短静默相同，选断页序号字典序最小（更早翻页）。
    plan = plan_pages(
        heights=[1] * 4,
        silences=[5, 5, 5, 0],
        page_height=3,
        min_turn_time=5,
    )
    assert plan.page_count == 2
    assert plan.breaks == [1]
    assert plan.shortest_turn_silence == 5

    # 多断页并列：容量 2 的 7 小节，所有内部静默相同，共需 4 页且有 1 小节余量。
    # 可行断页序列 [1,3,5]/[2,3,5]/[2,4,5]/[2,4,6]，字典序最小为 [1,3,5]。
    plan2 = plan_pages(
        heights=[1] * 7,
        silences=[5] * 6 + [0],
        page_height=2,
        min_turn_time=5,
    )
    assert plan2.page_count == 4
    assert plan2.breaks == [1, 3, 5]
    assert plan2.shortest_turn_silence == 5


def test_measure_too_tall():
    with pytest.raises(PlanningError, match="超过页面可用高度"):
        plan_pages(
            heights=[5, 8],
            silences=[1, 0],
            page_height=6,
            min_turn_time=1,
        )


def test_no_valid_plan():
    # 两小节各高 5、容量 6 -> 必须分开，但小节后静默 0 不达标。
    with pytest.raises(PlanningError, match="无合法翻页方案"):
        plan_pages(
            heights=[5, 5],
            silences=[0, 0],
            page_height=6,
            min_turn_time=1,
        )


def test_float_thresholds():
    # 容量 2、4 小节：2 页方案唯一断页 b=2，翻页静默 = s[1] = 1.5。
    # b=1 的方案需 3 页（页数更多，不优）。
    plan = plan_pages(
        heights=[1.0] * 4,
        silences=[2.5, 1.5, 2.5, 0],
        page_height=2.0,
        min_turn_time=1.5,
    )
    assert plan.page_count == 2
    assert plan.shortest_turn_silence == 1.5
    assert plan.breaks == [2]


def test_exact_fill_page_boundary():
    # 页恰好装满：高度等于 page_height 必须允许（<= 判定）；两个方案等价时字典序优先
    plan = plan_pages(
        heights=[3, 3, 3],
        silences=[2, 2, 0],
        page_height=6,
        min_turn_time=2,
    )
    assert plan.page_count == 2
    assert plan.breaks == [1]
    assert _ranges(plan) == [(0, 0), (1, 2)]


def test_large_scale_performance():
    # 10 万小节、页容 50：恰好 2000 页，所有翻页静默为 3 秒，1 秒级完成。
    n = 100_000
    plan = plan_pages([1] * n, [3.0] * n, 50, 2)
    assert plan.page_count == 2000
    assert plan.shortest_turn_silence == 3.0
    assert all(p.used_height <= 50 + 1e-9 for p in plan.pages)


@pytest.mark.parametrize("seed", range(200))
def test_matches_bruteforce(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    if seed % 2 == 0:
        heights = [rng.randint(1, 5) for _ in range(n)]
    else:  # 浮点高度
        heights = [rng.choice([0.5, 1.25, 1.5, 2.75, 3.0]) for _ in range(n)]
    silences = [rng.choice([0.0, 1.0, 5.0, 9.0]) for _ in range(n)]
    page_height = rng.choice([1, 2, 3, 5, 6, 10]) + (0.5 if seed % 2 else 0)
    min_turn_time = rng.choice([0.0, 1.0, 5.0])

    too_tall = any(h > page_height for h in heights)
    expected = brute_force_best(heights, silences, page_height, min_turn_time)

    if too_tall:
        with pytest.raises(PlanningError, match="超过页面可用高度"):
            plan_pages(heights, silences, page_height, min_turn_time)
        return

    if expected is None:
        with pytest.raises(PlanningError, match="无合法翻页方案"):
            plan_pages(heights, silences, page_height, min_turn_time)
    else:
        plan = plan_pages(heights, silences, page_height, min_turn_time)
        assert plan.breaks == expected, (
            f"mismatch h={heights} s={silences} H={page_height} t={min_turn_time}"
        )
        # 页高度均不超限
        for p in plan.pages:
            assert p.used_height <= page_height + 1e-9
        # 断页静默均达标
        for b in plan.breaks:
            assert silences[b - 1] + 1e-9 >= min_turn_time
