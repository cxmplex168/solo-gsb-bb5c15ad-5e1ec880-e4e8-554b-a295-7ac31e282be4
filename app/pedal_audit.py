"""排练踏板日志校对核心。

背景
====
``rehearsal`` 模块已能按排练参数（每小节拍数、速度、互不相交的重复段）生成
**应翻页提示**：仅当实际演奏序列中相邻两次发声跨页时给出一条提示，因此天然
包含重复段回跳造成的回翻提示。实际排练时，演奏者踩下翻页踏板会留下一条日志：
若干按时间非降序排列的踏板事件 ``(秒数, 操作后页码)``。本模块把日志与应翻页
提示做一对一校对：

- 一次踏板事件与一条提示**匹配**：事件操作后页码等于提示目标页（``to_page``），
  且二者时间差不超过非负时间容差；
- 没有对应事件的提示记为**漏翻**（该翻时没踩踏板）；
- 没有对应提示的事件记为**多翻**（踩了多余的踏板）。

按事件顺序做全局对齐（保序的一对一匹配，类似带容差的序列对齐），优化目标依次为：

1. **漏翻与多翻总数最少**（即匹配数最多）；
2. 再最小化**全部匹配的时间差绝对值之和**；
3. 仍并列时取**匹配的踏板事件序号序列字典序最小**；若连该序列也相同（同样的
   事件子集分别配到不同提示），则优先把较早的提示配成匹配（匹配的提示序号序列
   字典序最小），使结果完全确定、可复现。

实现上只对含可匹配事件的"活动提示行"做后缀标量 DP（相邻活动行之间只能整段
漏翻，批量跳过；每行只覆盖未来可达的事件下标走廊），再以显式栈的后序记忆化
搜索在最优代价上重构字典序最小的匹配序列；走廊之外的状态用闭式公式直接回答。

本模块仅依赖标准库，便于脱离 Web 层直接单测。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: 踏板事件数上限（与纸质模块 500 页上限保持同一量级）
MAX_EVENTS = 500

#: 标量比较时随数量级缩放的浮点容差倍数
_EPS_SCALE = 1e-9


class PedalAuditError(ValueError):
    """踏板日志输入不合法：事件乱序、秒数越界、页码越界、容差为负等。"""


@dataclass(frozen=True)
class PedalEvent:
    """一条踏板日志事件。

    :param seconds: 踩踏板时刻（秒，以演奏开始为 0），必须落在
        ``[0, 演奏总时长]`` 内
    :param page_after: 踩踏板操作后的页码（1 基），必须落在规划页码范围内
    """

    seconds: float
    page_after: int


@dataclass(frozen=True)
class CueMatch:
    """一条应翻页提示的校对结果。

    :param cue_index: 提示序号（0 基，按边界时刻升序，与输入提示一一对应）
    :param boundary_seconds: 提示的翻页边界时刻（秒）
    :param to_page: 提示要求翻到的目标页码（1 基）
    :param matched: 是否有踏板事件与该提示匹配
    :param event_index: 匹配的踏板事件序号（0 基）；漏翻时为 None
    :param event_seconds: 匹配事件的踩踏板时刻（秒）；漏翻时为 None
    :param time_diff_seconds: 匹配时间差（事件秒数 − 边界秒数，保留正负号
        以便看出早踩/晚踩）；漏翻时为 None
    """

    cue_index: int
    boundary_seconds: float
    to_page: int
    matched: bool
    event_index: int | None
    event_seconds: float | None
    time_diff_seconds: float | None


@dataclass(frozen=True)
class EventMark:
    """一条踏板事件的多翻标记。

    :param event_index: 踏板事件序号（0 基，按输入顺序）
    :param seconds: 踩踏板时刻（秒）
    :param page_after: 操作后页码（1 基）
    :param extra: 是否为多翻（未与任何提示匹配）
    :param cue_index: 与之匹配的提示序号（0 基）；多翻时为 None
    """

    event_index: int
    seconds: float
    page_after: int
    extra: bool
    cue_index: int | None


@dataclass(frozen=True)
class PedalAuditResult:
    """踏板日志校对结果。"""

    total_duration: float  # 按重复段展开后的演奏总时长（秒）
    tolerance_seconds: float  # 请求的时间容差（秒）
    cue_results: List[CueMatch]  # 每条应翻页提示的匹配/漏翻结果
    event_marks: List[EventMark]  # 每条踏板事件的多翻标记
    missed_count: int  # 漏翻条数（未匹配的提示数）
    extra_count: int  # 多翻条数（未匹配的事件数）
    error_count: int  # 漏翻与多翻总数
    total_time_diff: float  # 全部匹配的时间差绝对值之和（秒）


def _eps(magnitude: float) -> float:
    """随数量级缩放的浮点容差，抹平累加与 JSON 浮点解析误差。"""
    return _EPS_SCALE * max(1.0, abs(magnitude))


def _normalize_events(
    events: Sequence[PedalEvent | Tuple[float, int]],
    total_duration: float,
    page_count: int,
) -> List[PedalEvent]:
    """校验踏板事件：秒数有限非负且不越演奏时长、页码严格整数且不越页范围、
    时间非降序。"""
    if len(events) > MAX_EVENTS:
        raise PedalAuditError(
            f"踏板事件数（{len(events)}）超过上限 {MAX_EVENTS}，无法校对"
        )

    checked: List[PedalEvent] = []
    for idx, event in enumerate(events):
        if isinstance(event, PedalEvent):
            seconds, page = event.seconds, event.page_after
        else:
            try:
                seconds, page = event
            except (TypeError, ValueError) as exc:
                raise PedalAuditError(
                    f"第 {idx + 1} 个踏板事件必须是 (秒数, 操作后页码) 二元组"
                ) from exc
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件的秒数必须是数值（收到 {seconds!r}）"
            )
        seconds_f = float(seconds)
        if seconds_f != seconds_f or seconds_f in (float("inf"), float("-inf")):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件的秒数必须是有限数值"
            )
        if seconds_f < 0:
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件秒数 {seconds_f:g} 不能为负数"
                "（以演奏开始为 0）"
            )
        if seconds_f > total_duration + _eps(total_duration):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件秒数 {seconds_f:g} 超出演奏总时长"
                f" {total_duration:g} 秒（含端点）"
            )
        if not isinstance(page, int) or isinstance(page, bool):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件的操作后页码必须为整数"
                f"（收到 {page!r}），不接受布尔值或小数"
            )
        if page < 1 or page > page_count:
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件的操作后页码 {page} 越界："
                f"合法范围为 1..{page_count}"
            )
        checked.append(PedalEvent(seconds_f, page))

    for idx in range(1, len(checked)):
        prev_t, cur_t = checked[idx - 1].seconds, checked[idx].seconds
        if cur_t < prev_t - _eps(prev_t):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件秒数 {cur_t:g} 早于前一事件"
                f" {prev_t:g}：踏板事件必须按时间非降序排列"
            )
    return checked


def audit_pedal_log(
    cue_boundaries: Sequence[float],
    cue_target_pages: Sequence[int],
    events: Sequence[PedalEvent | Tuple[float, int]],
    total_duration: float,
    page_count: int,
    tolerance_seconds: float,
) -> PedalAuditResult:
    """把踏板日志与应翻页提示做保序一对一校对。

    :param cue_boundaries: 各提示的翻页边界时刻（秒，升序）
    :param cue_target_pages: 各提示的目标页码（1 基，与 ``cue_boundaries``
        等长）；匹配要求事件操作后页码与之相等
    :param events: 踏板事件（秒数, 操作后页码），按时间非降序
    :param total_duration: 演奏总时长（秒），事件秒数不得超过该值
    :param page_count: 规划页数，事件页码不得越出 ``1..page_count``
    :param tolerance_seconds: 非负时间容差（秒），时间差不超过该值才允许匹配
    :return: 每条提示的匹配/漏翻、每条事件的多翻标记与两项代价
    :raises PedalAuditError: 任一输入约束不满足
    """
    if len(cue_boundaries) != len(cue_target_pages):
        raise PedalAuditError("提示边界时刻与目标页码数组长度不一致")
    if not isinstance(tolerance_seconds, (int, float)) or isinstance(
        tolerance_seconds, bool
    ):
        raise PedalAuditError("时间容差必须是数值")
    tol = float(tolerance_seconds)
    if tol != tol or tol in (float("inf"), float("-inf")):
        raise PedalAuditError("时间容差必须是有限数值")
    if tol < 0:
        raise PedalAuditError(
            f"时间容差不能为负数（实际 {tol:g}）"
        )
    if not isinstance(total_duration, (int, float)) or total_duration < 0:
        raise PedalAuditError("演奏总时长必须为非负数值")

    checked_events = _normalize_events(events, float(total_duration), page_count)
    n_cues = len(cue_boundaries)
    n_events = len(checked_events)
    event_times = [e.seconds for e in checked_events]
    event_pages = [e.page_after for e in checked_events]

    # 每个提示行的可匹配事件下标（页码相同且 |时间差| ≤ tol），按 j 升序。
    # 只保留至少有一个候选事件的"活动行"。
    cand_by_cue: List[List[int]] = [[] for _ in range(n_cues)]
    active: List[int] = []
    for i, (boundary, target_page) in enumerate(
        zip(cue_boundaries, cue_target_pages)
    ):
        b = float(boundary)
        left = bisect_left(event_times, b - tol - _eps(b))
        right = bisect_right(event_times, b + tol + _eps(b))
        cands = [
            j
            for j in range(left, right)
            if event_pages[j] == target_page
            and abs(event_times[j] - b) <= tol + _eps(b)
        ]
        if cands:
            cand_by_cue[i] = cands
            active.append(i)

    n_active = len(active)
    # hi[p]：活动行 p 及之后所有行候选事件的最大下标（无候选记 -1）；
    # lo[p]：活动行 p 及之后所有行候选事件的最小下标（无候选记 n_events）。
    # 后缀表 F(p, j) 只需落 j ∈ [lo[p], hi[p]+1]：j < lo[p] 时中间事件只能
    # 多翻，与 (p, lo[p]) 同形；j > hi[p] 时当前及以后都不可能再匹配，闭式
    # 回答；hi[p]+1 是表内终端列。
    hi: List[int] = [-1] * (n_active + 1)
    lo: List[int] = [n_events] * (n_active + 1)
    for p in range(n_active - 1, -1, -1):
        cands = cand_by_cue[active[p]]
        hi[p] = max(hi[p + 1], cands[-1])
        lo[p] = min(lo[p + 1], cands[0])

    def _tail(p: int, j: int) -> Tuple[int, float]:
        """活动行 p..末行整体不再匹配后的尾部标量代价 (漏翻+多翻, 时间差和)。"""
        missed = n_cues - (active[p] if p < n_active else n_cues)
        extras = max(0, n_events - j)
        return missed + extras, 0.0

    # rows[p][j - lo[p]] = F(p, j)，j ∈ [lo[p], hi[p]+1]；末列是"本行及
    # 以后最后一个候选事件也已越过"的终端态（闭式）。按 p 递减建表，行内
    # 按 j 递减填表（多翻转移 F(p, j+1) 同行走右列，须先填）。
    rows: List[List[Tuple[int, float] | None]] = [
        [None] * (hi[p] + 1 - lo[p] + 1) for p in range(n_active)
    ]

    def _value(p: int, j: int) -> Tuple[int, float]:
        """后缀状态 F(活动行 p, 下一事件下标 j) 的标量代价。

        走廊之外闭式回答：j > hi[p] 时当前及以后不可能再匹配；j < lo[p]
        时事件 j..lo[p)-1 只能逐条多翻，代价等于 F(p, lo[p]) 加上多翻数。
        """
        if p >= n_active:
            return _tail(p, j)
        if j > hi[p]:
            return _tail(p, j)
        if j < lo[p]:
            base = _value(p, lo[p])
            return base[0] + (lo[p] - j), base[1]
        value = rows[p][j - lo[p]]
        # 行内按 j 递减填表时，右列（更大的 j）必先填好
        assert value is not None, f"F({p}, {j}) 尚未填充"
        return value

    # 自最后一个活动行向前做后缀标量 DP。相邻活动行 i_p、i_{p+1} 之间的提示
    # 全部漏翻，故从行 p 走到行 p+1 固定付出 gap 次漏翻。
    for p in range(n_active - 1, -1, -1):
        i = active[p]
        cands = cand_by_cue[i]
        b = float(cue_boundaries[i])
        gap = (active[p + 1] - i - 1) if p + 1 < n_active else (n_cues - 1 - i)
        row = rows[p]
        for j in range(hi[p] + 1, lo[p] - 1, -1):
            if j > hi[p]:
                # 终端列：本行及以后都不可能再匹配，整段漏翻 + 剩余事件多翻
                row[j - lo[p]] = _tail(p, j)
                continue
            # 多翻事件 j（计 1 次多翻）：1 + F(p, j+1)（右列已填，含终端列）
            extra_tail = _value(p, j + 1)
            best = (extra_tail[0] + 1, extra_tail[1])
            # 本提示漏翻（连同与下一活动行之间的 gap 个漏翻）
            miss_tail = _value(p + 1, j)
            miss_cost = (miss_tail[0] + 1 + gap, miss_tail[1])
            if miss_cost < best:
                best = miss_cost
            # 匹配：仅考虑页码相同且容差内的候选事件 k ≥ j。事件 j..k-1 在
            # 此匹配前只能多翻（k-j 次），提示 i+1..下一活动行-1 全部漏翻
            # （gap 次），尾部从 F(p+1, k+1) 接续。
            for k in cands:
                if k < j:
                    continue
                tail = _value(p + 1, k + 1)
                diff = abs(event_times[k] - b)
                match_cost = (tail[0] + (k - j) + gap, tail[1] + diff)
                if match_cost < best:
                    best = match_cost
            row[j - lo[p]] = best

    # 顶层标量代价：第一个活动行之前的提示（0..active[0)-1）没有任何候选
    # 事件，必然全部漏翻；F(0, 0) 只覆盖 active[0] 及其之后的提示。
    if n_active:
        top_value = (
            _value(0, 0)[0] + active[0],
            _value(0, 0)[1],
        )
    else:
        top_value = (n_cues + n_events, 0.0)
    top_error, top_diff = top_value

    # 在最优标量代价上重构字典序最小的匹配序列。显式栈做两阶段（visit /
    # finalize）后序记忆化，避免活动行很多时递归过深，也保证共享依赖在
    # finalize 时必已入 memo（依赖先于 finalize 标记入栈）。记忆化状态一律
    # 规整到走廊内：j < lo[p] 与 (p, lo[p]) 同键（之间的事件只能多翻，不改
    # 匹配序列），j > hi[p]（含 p == n_active）为终端态。每个状态至多三类
    # 候选动作：多翻事件 j、本行漏翻、与最小的"最优可行"候选事件 k 匹配——
    # 匹配分支的键以 k 开头，故只需考察取得最优代价的最小 k。
    SeqPair = Tuple[Tuple[int, ...], Tuple[int, ...]]
    EMPTY: SeqPair = ((), ())
    memo: Dict[Tuple[int, int], SeqPair] = {}

    def _gaps(p: int) -> int:
        return (
            (active[p + 1] - active[p] - 1)
            if p + 1 < n_active
            else n_cues - 1 - active[p]
        )

    def _best_match_k(p: int, j: int) -> int | None:
        """行 p 在状态 j 下取得最优代价的最小匹配候选 k（无则 None）。"""
        i = active[p]
        b = float(cue_boundaries[i])
        gap = _gaps(p)
        cur_value = _value(p, j)
        for k in cand_by_cue[i]:
            if k < j:
                continue
            tail = _value(p + 1, k + 1)
            cost = (
                tail[0] + (k - j) + gap,
                tail[1] + abs(event_times[k] - b),
            )
            if cost[0] == cur_value[0] and abs(cost[1] - cur_value[1]) <= (
                _eps(cur_value[1]) + 1e-9
            ):
                return k  # 候选按 k 升序，第一个即最小
        return None

    def _norm(p: int, j: int) -> Tuple[int, int]:
        """把状态规整到走廊规范键：终端态返回 (-1, -1)，下界归一到 lo[p]。"""
        if p >= n_active or j > hi[p]:
            return (-1, -1)
        if j < lo[p]:
            return (p, lo[p])
        return (p, j)

    # 栈帧：(p, j, finalize)；finalize=False 为首次访问，True 为依赖齐备后求值。
    # 栈内一律存放规范走廊状态（或终端态）。
    stack: List[Tuple[int, int, bool]] = [(0, max(0, lo[0]), False)]
    while stack:
        p, j, finalize = stack.pop()
        state = (p, j)
        if not finalize:
            if state == (-1, -1) or state in memo:
                continue
            if p >= n_active or j > hi[p]:
                memo[state] = EMPTY
                continue
            # 先压 finalize 标记，再按依赖逆序压栈：依赖全部解析后标记才弹出
            stack.append((p, j, True))
            # 本行漏翻 -> (p+1, j)
            stack.append((*_norm(p + 1, j), False))
            # 最小的最优匹配候选 k -> (p+1, k+1)
            chosen_k = _best_match_k(p, j)
            if chosen_k is not None:
                stack.append((*_norm(p + 1, chosen_k + 1), False))
            # 多翻事件 j -> (p, j+1)
            stack.append((*_norm(p, j + 1), False))
            continue

        if state in memo:
            continue
        i = active[p]
        b = float(cue_boundaries[i])
        gap = _gaps(p)
        cur_value = _value(p, j)
        best_key: SeqPair | None = None

        def _consider(cost: Tuple[int, float], key: SeqPair) -> None:
            nonlocal best_key
            optimal = cost[0] == cur_value[0] and abs(
                cost[1] - cur_value[1]
            ) <= _eps(cur_value[1]) + 1e-9
            if optimal and (best_key is None or key < best_key):
                best_key = key

        # 多翻事件 j（计 1 次；依赖键 (p, j+1) 的规范形式，终端态为空序列）
        extra_value = _value(p, j + 1)
        extra_key = memo.get(_norm(p, j + 1), EMPTY)
        _consider((extra_value[0] + 1, extra_value[1]), extra_key)

        # 本行漏翻（连同与下一活动行之间的 gap 个漏翻）
        miss_key = memo.get(_norm(p + 1, j), EMPTY)
        miss_value = _value(p + 1, j)
        _consider((miss_value[0] + 1 + gap, miss_value[1]), miss_key)

        # 匹配最小的最优候选 k；更大的 k 首元素更大，不可能更优
        chosen_k = _best_match_k(p, j)
        if chosen_k is not None:
            tail_key = memo.get(_norm(p + 1, chosen_k + 1), EMPTY)
            tail_value = _value(p + 1, chosen_k + 1)
            _consider(
                (
                    tail_value[0] + (chosen_k - j) + gap,
                    tail_value[1] + abs(event_times[chosen_k] - b),
                ),
                ((chosen_k,) + tail_key[0], (i,) + tail_key[1]),
            )

        assert best_key is not None
        memo[state] = best_key

    if n_active:
        matched_events, matched_cues = memo[(0, max(0, lo[0]))]
    else:
        matched_events, matched_cues = EMPTY

    event_to_cue = {j: i for j, i in zip(matched_events, matched_cues)}

    cue_results: List[CueMatch] = []
    cue_to_event = {i: j for j, i in event_to_cue.items()}
    total_diff = 0.0
    missed = 0
    for i in range(n_cues):
        boundary = float(cue_boundaries[i])
        j = cue_to_event.get(i)
        if j is None:
            missed += 1
            cue_results.append(
                CueMatch(
                    cue_index=i,
                    boundary_seconds=boundary,
                    to_page=int(cue_target_pages[i]),
                    matched=False,
                    event_index=None,
                    event_seconds=None,
                    time_diff_seconds=None,
                )
            )
        else:
            diff = round(event_times[j] - boundary, 9)
            if diff == 0:
                diff = 0.0
            total_diff += abs(diff)
            cue_results.append(
                CueMatch(
                    cue_index=i,
                    boundary_seconds=boundary,
                    to_page=int(cue_target_pages[i]),
                    matched=True,
                    event_index=j,
                    event_seconds=event_times[j],
                    time_diff_seconds=diff,
                )
            )

    event_marks: List[EventMark] = []
    extra = 0
    for j, event in enumerate(checked_events):
        matched_cue = event_to_cue.get(j)
        if matched_cue is None:
            extra += 1
        event_marks.append(
            EventMark(
                event_index=j,
                seconds=event.seconds,
                page_after=event.page_after,
                extra=matched_cue is None,
                cue_index=matched_cue,
            )
        )

    total_diff = round(total_diff, 9)
    if total_diff == 0:
        total_diff = 0.0
    # 与 DP 顶层代价互校
    assert missed + extra == top_error
    assert abs(total_diff - top_diff) <= _eps(top_diff) + 1e-9

    return PedalAuditResult(
        total_duration=float(total_duration),
        tolerance_seconds=tol,
        cue_results=cue_results,
        event_marks=event_marks,
        missed_count=missed,
        extra_count=extra,
        error_count=missed + extra,
        total_time_diff=total_diff,
    )
