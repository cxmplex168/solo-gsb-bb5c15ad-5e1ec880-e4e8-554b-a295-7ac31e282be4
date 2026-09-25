"""翻页点规划动态规划核心。

问题建模
========
小节按顺序编号 0..n-1，每小节高度 ``heights[i]``，第 i 小节之后的静默时间为
``silences[i]``（单位：秒）。一页可容纳连续若干小节，且总高度不超过
``page_height``。除最后一页外，每个断页位置 b（表示第 b-1 小节之后翻页，
即下一页从第 b 小节开始）必须满足 ``silences[b-1] >= min_turn_time``。

目标（按优先级）：
1. 最小化总页数；
2. 在页数最少的方案中，最大化所有翻页点静默时间的最小值；
3. 仍并列时，选择断页序号序列字典序最小的方案。

三个目标均在**全篇统一比较**：必翻/禁翻约束只改变合法断页位置集合，
不会把乐谱按必翻点切成若干段各自取局部最优。

可选约束（``required_breaks`` / ``forbidden_breaks``，元素为新页起始小节的
0 基序号，合法范围 1..n-1）：
- 必翻位置一定是断页（其静默还必须达到最低翻页时间，否则无解）；
- 禁翻位置一定不是断页；
- 两数组各自重复、互相交叉，或含 0、末尾序号 n 及越界值，均判定为输入非法。

求解分三步（均为动态规划/基于 DP 的贪心重构）：
1. 前向最短路 DP 求最小页数 K；必翻点通过页首下界强制断页、禁翻点从允许
   位置集合剔除；不可达即无合法方案；
2. 对"所有翻页点静默 >= t"这一可行性判定做二分（判定仍是同一条最短路 DP，
   只是允许的断页位置减少），求最大可行 t；
3. 在阈值 t 下做后缀 min/max 页数 DP（必翻点通过页尾上界强制），贪心选择
   最小的合法断页序号，得到字典序最小的断页序列。

单小节高度超过页面可用高度时无法排版，直接报错。
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

INF = 10**15


def _eps(magnitude: float) -> float:
    """随高度量级缩放的浮点容差，避免长序列前缀和的累加误差导致误判。"""
    return 1e-9 * max(1.0, abs(magnitude))


class PlanningError(ValueError):
    """规划失败：输入语义不合法（单小节超高、无合法方案等）。"""


@dataclass(frozen=True)
class Page:
    """一页的排版结果。小节序号均为 0 基、端点闭区间。"""

    page_index: int
    start_measure: int  # 起始小节（0 基）
    end_measure: int  # 结束小节（0 基，闭区间）
    measure_count: int
    used_height: float
    # 本页之后翻页点的静默秒数；最后一页无翻页点，为 None
    turn_silence: Optional[float]


@dataclass(frozen=True)
class Plan:
    pages: List[Page]
    page_count: int
    # 所有翻页点中最短的静默秒数；单页（无翻页点）时为 None
    shortest_turn_silence: Optional[float]
    # 断页序号序列：每个元素是"新一页起始小节"的 0 基序号
    breaks: List[int]


def _page_end_range(prefix: Sequence[float], r: int, page_height: float) -> tuple[int, int]:
    """以第 r 个小节（0 基）为页尾时，页首 l（0 基）的取值范围。

    页内容为小节 l..r，高度为 prefix[r+1] - prefix[l] <= page_height，
    即 prefix[l] >= prefix[r+1] - page_height。
    """
    threshold = prefix[r + 1] - page_height
    lo = bisect_left(prefix, threshold - _eps(threshold), 0, r + 1)
    return lo, r


def _min_pages_forward(
    n: int,
    prefix: Sequence[float],
    page_height: float,
    allowed: Callable[[int], bool],
    floor_forced: Optional[Sequence[int]] = None,
) -> List[int]:
    """前向 DP：f[r] = 覆盖小节 0..r 所需的最少页数。

    一页小节 l..r 合法当且仅当高度不超限，且 l == 0 或断页位置 l 允许翻页。
    滑动窗口（l 的可行区间随 r 单调）用单调队列维护 f[l-1] 的最小值，
    整体 O(n)。

    ``floor_forced[r]`` 给出以 r 为页尾时页首 l 的下界：<= r 的最后一个必翻点
    （0 表示无）。必翻点 q 必须恰好是某页页首，因此含 q 的页只能以 q 起始，
    强制 l >= floor_forced[r]。
    """
    if floor_forced is None:
        floor_forced = [0] * n
    # 状态以"已覆盖小节数 m"表示：g[m] = f[m-1]，g[0] = 0（空前缀）。
    g: List[int] = [INF] * (n + 1)
    g[0] = 0
    dq: deque[int] = deque()
    for r in range(n):
        m = r  # 新候选：页 [r..r]，对应前缀状态 g[r]
        # 在 r 处断页意味着下一页从第 r 小节开始，r>0 时必须允许翻页。
        if g[m] < INF and (m == 0 or allowed(m)):
            while dq and g[dq[-1]] >= g[m]:
                dq.pop()
            dq.append(m)
        lo, _ = _page_end_range(prefix, r, page_height)
        # 必翻点下界同样可能高于高度窗口下界，取两者较大者。
        lo = max(lo, floor_forced[r])
        while dq and dq[0] < lo:
            dq.popleft()
        g[r + 1] = g[dq[0]] + 1 if dq else INF
    return [g[r + 1] for r in range(n)]


def _suffix_min_max_pages(
    n: int,
    prefix: Sequence[float],
    page_height: float,
    allowed: Callable[[int], bool],
    next_forced: Optional[Sequence[int]] = None,
) -> tuple[List[int], List[int]]:
    """后缀 DP：从第 i 小节开始排完剩余内容所需的最少/最多页数。

    允许页数在 [smin[i], smax[i]] 之间连续可取（任意合法方案都能在允许翻页的
    位置把一页继续拆成两页，从而逐页 +1，故中间页数均可达；必翻点只是把部分
    拆分位置改成强制，区间连续性不受影响）。
    窗口随 i 反向单调，用两个单调队列分别维护最小、最大值，整体 O(n)。

    ``next_forced[i]`` 为严格大于 i 的第一个必翻点（不存在时为 n）：以 i 为
    页首的页不能越过该必翻点，故页尾上界 e <= next_forced[i]，必翻点因此
    强制成为下一页页首。
    """
    if next_forced is None:
        next_forced = [n] * (n + 1)
    smin: List[int] = [INF] * (n + 1)
    smax: List[int] = [-INF] * (n + 1)
    smin[n] = 0
    smax[n] = 0

    min_dq: deque[int] = deque()  # 按 smin 值升序
    max_dq: deque[int] = deque()  # 按 smax 值降序

    for i in range(n - 1, -1, -1):
        # 上一轮窗口里 e == i 的候选已过期（页必须至少包含第 i 小节，e > i）。
        # 索引从右（最新、最小）加入，故从右端剔除 e <= i。
        while min_dq and min_dq[-1] <= i:
            min_dq.pop()
        while max_dq and max_dq[-1] <= i:
            max_dq.pop()
        # 新候选 e = i+1：页 [i..i] 后接后缀 i+1。断页位置 i+1 必须允许翻页，
        # 除非 i+1 == n（本页即最后一页）。e = i+1 必不越过下一个必翻点。
        e = i + 1
        if e == n or allowed(e):
            if smin[e] < INF:
                while min_dq and smin[min_dq[-1]] >= smin[e]:
                    min_dq.pop()
                min_dq.append(e)
            if smax[e] > -INF:
                while max_dq and smax[max_dq[-1]] <= smax[e]:
                    max_dq.pop()
                max_dq.append(e)
        # 页 [i..e-1] 高度 = prefix[e] - prefix[i] <= page_height，
        # 即 prefix[e] <= prefix[i] + page_height；bisect_left 给出第一个
        # 超限的索引，减 1 得最大合法 e。
        max_height = prefix[i] + page_height
        max_e = (
            bisect_left(prefix, max_height + _eps(max_height), i + 1, n + 1) - 1
        )
        # 必翻点上界：本页不得越过 i 之后的第一个必翻点。
        max_e = min(max_e, next_forced[i])
        while min_dq and min_dq[0] > max_e:
            min_dq.popleft()
        while max_dq and max_dq[0] > max_e:
            max_dq.popleft()
        if min_dq:
            smin[i] = smin[min_dq[0]] + 1
        if max_dq:
            smax[i] = smax[max_dq[0]] + 1
    return smin, smax


def _validate_break_constraints(
    n: int,
    required_breaks: Sequence[int],
    forbidden_breaks: Sequence[int],
) -> tuple[List[int], frozenset[int]]:
    """校验必翻/禁翻序号：元素须为 int，范围 1..n-1，各自不重复、互不交叉。

    :raises PlanningError: 任一结构约束不满足。
    """
    required: List[int] = []
    seen: set[int] = set()
    for b in required_breaks:
        if not isinstance(b, int) or isinstance(b, bool):
            raise PlanningError(f"required_breaks 元素必须为整数序号（收到 {b!r}）。")
        if b < 1 or b > n - 1:
            raise PlanningError(
                f"required_breaks 含越界序号 {b}：合法范围为 1..{n - 1}"
                "（0 是首页起点、末尾序号代表文档结束，均不能作为断页位置）。"
            )
        if b in seen:
            raise PlanningError(f"required_breaks 含重复序号 {b}。")
        seen.add(b)
        required.append(b)
    required.sort()

    forbidden: set[int] = set()
    for b in forbidden_breaks:
        if not isinstance(b, int) or isinstance(b, bool):
            raise PlanningError(f"forbidden_breaks 元素必须为整数序号（收到 {b!r}）。")
        if b < 1 or b > n - 1:
            raise PlanningError(
                f"forbidden_breaks 含越界序号 {b}：合法范围为 1..{n - 1}"
                "（0 是首页起点、末尾序号代表文档结束，均不能作为断页位置）。"
            )
        if b in forbidden:
            raise PlanningError(f"forbidden_breaks 含重复序号 {b}。")
        forbidden.add(b)

    clash = sorted(seen & forbidden)
    if clash:
        raise PlanningError(
            f"序号 {clash[0]} 同时出现在 required_breaks 与 forbidden_breaks 中，"
            "必翻与禁翻约束不能交叉。"
        )
    return required, frozenset(forbidden)


def plan_pages(
    heights: Sequence[float],
    silences: Sequence[float],
    page_height: float,
    min_turn_time: float,
    required_breaks: Sequence[int] = (),
    forbidden_breaks: Sequence[int] = (),
) -> Plan:
    """计算最优翻页方案。

    ``required_breaks`` 为必翻位置（新页起始小节的 0 基序号），必须恰好成为
    断页；``forbidden_breaks`` 为禁翻位置，必须恰好不是断页。三个优化目标
    （页数、最短翻页静默、断页序列字典序）始终在全篇所有满足约束的方案中
    统一比较，不按必翻点分段取局部最优。

    :raises PlanningError: 约束序号非法、必翻点静默不足、单小节超高，
        或不存在任何合法断页方案。
    """
    n = len(heights)

    for i, h in enumerate(heights):
        if h > page_height + _eps(page_height):
            raise PlanningError(
                f"第 {i + 1} 小节高度 {h:g} 超过页面可用高度 {page_height:g}，"
                "小节不可拆开，无法排版。"
            )

    required, forbidden = _validate_break_constraints(
        n, required_breaks, forbidden_breaks
    )

    # 必翻位置本身必须是合法翻页点：其后静默不足则不存在满足约束的方案。
    quiet_required = [
        b
        for b in required
        if silences[b - 1] + _eps(min_turn_time) < min_turn_time
    ]
    if quiet_required:
        b = quiet_required[0]
        raise PlanningError(
            f"必翻位置 {b}（第 {b} 小节之后）静默仅 {silences[b - 1]:g} 秒，"
            f"低于最低翻页时间 {min_turn_time:g} 秒，必翻点静默不足，无合法方案。"
        )

    prefix: List[float] = [0.0]
    for h in heights:
        prefix.append(prefix[-1] + h)

    # 断页位置 b（下一页从第 b 小节开始，0 基）：
    # b == 0 是起点占位；内部翻页看第 b-1 小节后的静默；b == n 为文档结束，
    # 不作为翻页点。禁翻位置恒不允许；必翻位置若静默达标（已前置校验）则允许。
    def baseline_allowed(b: int) -> bool:
        if b <= 0 or b >= n or b in forbidden:
            return False
        return silences[b - 1] + _eps(min_turn_time) >= min_turn_time

    # floor_forced[r]：<= r 的最后一个必翻点（0 表示无）。页 [l..r] 内含必翻点
    # q（l <= q <= r）时唯一合法情形是 l == q，即 l >= floor_forced[r]。
    floor_forced: List[int] = [0] * n
    cursor = 0
    last = 0
    for r in range(n):
        while cursor < len(required) and required[cursor] <= r:
            last = required[cursor]
            cursor += 1
        floor_forced[r] = last

    f = _min_pages_forward(n, prefix, page_height, baseline_allowed, floor_forced)
    total = f[n - 1]
    if total >= INF:
        raise PlanningError(
            "无合法翻页方案：在满足最低翻页时间、必翻/禁翻约束的位置无法完成"
            "分页（存在小节无法与相邻小节排入同一页且其后静默不足，"
            "或必翻/禁翻位置相互冲突）。"
        )

    if total == 1:
        page = Page(
            page_index=1,
            start_measure=0,
            end_measure=n - 1,
            measure_count=n,
            used_height=prefix[n],
            turn_silence=None,
        )
        return Plan(pages=[page], page_count=1, shortest_turn_silence=None, breaks=[])

    # 下一步起需要"严格大于 i 的第一个必翻点"，正/反各做一次 O(n) 预处理。
    # next_forced[i]：以 i 为页首的页尾 e 不得越过该必翻点（e <= next_forced[i]），
    # 必翻点因此强制成为下一页页首。
    next_forced: List[int] = [n] * (n + 1)
    cursor = len(required) - 1
    nxt = n
    for i in range(n, -1, -1):
        while cursor >= 0 and required[cursor] > i:
            nxt = required[cursor]
            cursor -= 1
        next_forced[i] = nxt

    # ---- 第二步：最大化最短翻页静默（全篇统一，不按必翻点分段） ----
    candidates = sorted(
        {silences[b - 1] for b in range(1, n) if baseline_allowed(b)}
    )

    def feasible(threshold: float) -> bool:
        def allowed_t(b: int) -> bool:
            if b <= 0 or b >= n or b in forbidden:
                return False
            return silences[b - 1] + _eps(threshold) >= threshold

        ft = _min_pages_forward(
            n, prefix, page_height, allowed_t, floor_forced
        )
        return ft[n - 1] <= total

    lo, hi = 0, len(candidates) - 1  # candidates[lo] 必然可行（等于基线）
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(candidates[mid]):
            lo = mid
        else:
            hi = mid - 1
    best_threshold = candidates[lo]

    # ---- 第三步：阈值固定，贪心取字典序最小断页序列 ----
    def best_allowed(b: int) -> bool:
        if b <= 0 or b >= n or b in forbidden:
            return False
        return silences[b - 1] + _eps(best_threshold) >= best_threshold

    smin, smax = _suffix_min_max_pages(
        n, prefix, page_height, best_allowed, next_forced
    )

    breaks: List[int] = []
    start = 0
    remaining = total
    while remaining > 1:
        # 本页结束后下一断页 b（页 [start..b-1]），之后后缀用 remaining-1 页排完。
        max_height = prefix[start] + page_height
        max_b = (
            bisect_left(prefix, max_height + _eps(max_height), start + 1, n + 1) - 1
        )
        # 本页不得越过 start 之后的第一个必翻点：必翻点强制作为下一页页首，
        # 故候选 b 最大到 next_forced[start]（恰为该必翻点时即在此必翻）。
        max_b = min(max_b, next_forced[start])
        chosen: Optional[int] = None
        need = remaining - 1
        for b in range(start + 1, max_b + 1):
            if best_allowed(b) and smin[b] <= need <= smax[b]:
                chosen = b
                break
        if chosen is None:  # 理论上不可达：阈值方案已被证明可行
            raise PlanningError("内部错误：重构翻页方案失败。")
        breaks.append(chosen)
        start = chosen
        remaining -= 1

    if not set(required) <= set(breaks):  # 防御：必翻点必须全部出现
        raise PlanningError("内部错误：重构翻页方案遗漏必翻位置。")

    # ---- 组装页信息 ----
    boundaries = [0] + breaks + [n]
    pages: List[Page] = []
    turn_values: List[float] = []
    for idx in range(len(boundaries) - 1):
        a, e = boundaries[idx], boundaries[idx + 1]
        is_last = idx == len(boundaries) - 2
        silence = None if is_last else float(silences[e - 1])
        if silence is not None:
            turn_values.append(silence)
        pages.append(
            Page(
                page_index=idx + 1,
                start_measure=a,
                end_measure=e - 1,
                measure_count=e - a,
                used_height=prefix[e] - prefix[a],
                turn_silence=silence,
            )
        )

    shortest = min(turn_values) if turn_values else None
    return Plan(
        pages=pages,
        page_count=total,
        shortest_turn_silence=shortest,
        breaks=breaks,
    )
