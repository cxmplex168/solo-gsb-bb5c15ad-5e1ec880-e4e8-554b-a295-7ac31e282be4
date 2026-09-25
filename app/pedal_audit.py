"""排练踏板日志校对核心。

背景
====
排练时乐手按翻页提示踩踏板换页，踏板会记录一份日志：按时间排列的若干次踏板
操作，每次给出操作发生的秒数与**操作后页码**。演奏结束后，把这份日志与服务按
排练参数（每小节拍数、速度、重复段）生成的**应翻页提示**逐条校对：

- 每条提示都对应一次"应翻"；
- 一次踏板操作与一条提示**目标页相同**（提示的 ``to_page``）且**时间差不超过
  给定的非负时间容差**时，允许二者一对一匹配（时间差取绝对值）；
- 匹配按提示与日志各自的时间顺序做**全局对齐**（单调序列对齐），一条提示至多
  匹配一次踏板操作，反之亦然；
- 未匹配上的提示记为**漏翻**（应翻未翻），未匹配上的踏板操作记为**多翻**
  （无对应提示的多余操作）。

优化目标（依次比较）
====================

1. **漏翻数与多翻数之和最少**（即尽量多地匹配上）；
2. 上一项相同时，**所有匹配对的时间差之和最小**；
3. 仍并列时，取**匹配到的踏板事件序号序列字典序最小**的对齐（序号序列按提示
   顺序排列，因单调对齐天然严格递增；先比第一个匹配的事件序号，再依次比后续）。

算法
====

设提示 m 条、日志事件 n 条（均 ≤ 500）。先做后缀标量 DP：``G[i][j]`` /
``S[i][j]`` 分别为把提示 ``[i:]`` 与事件 ``[j:]`` 对齐的最小（漏翻+多翻）数与
该数下的最小时间差之和。三种转移：

- 漏翻提示 i：``(G[i+1][j]+1, S[i+1][j])``；
- 多翻事件 j：``(G[i][j+1]+1, S[i][j+1])``；
- 当目标页相同且 |t_i − t_j| ≤ 容差时匹配：``(G[i+1][j+1], S[i+1][j+1]+|Δt|)``。

字典序目标只用于同 ``(G, S)`` 的重构选择，且只影响被匹配的事件序号序列，不影响
两项代价值，故不把变长元组放进 O(m·n) 的标量表，而在前向重构时用记忆化递归
``best(i, j)`` 求该后缀在最优代价下的字典序最小匹配序号元组——每个格至多计算
一次，元组总长度 O(m·n·min(m,n)) 在 500 上限下约 17MB，可接受。

事件越出演奏时长（含恰等于总时长）或页码范围（不在 1..page_count）直接报错。

本模块仅依赖标准库，便于脱离 Web 层直接单测。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

# 对齐两侧（应翻提示数 / 踏板事件数）的共同上限，限制 O(m·n) 的 DP 规模
MAX_EVENTS = 500


class PedalAuditError(ValueError):
    """踏板日志校对输入不合法：事件越界、时间非降序破坏、长度超限等。"""


@dataclass(frozen=True)
class PedalEvent:
    """一次踏板操作日志记录。

    :param seconds: 操作发生时刻（秒，以演奏开始为 0）
    :param page_after: 操作完成后所在的 1 基页码
    """

    seconds: float
    page_after: int


@dataclass(frozen=True)
class CueAudit:
    """一条应翻页提示的校对结果。

    :param cue_index: 该提示在应翻页提示序列中的 0 基序号
    :param expected_seconds: 提示的边界时刻（秒）
    :param to_page: 提示要求翻到的目标页（1 基，也是匹配的目标页）
    :param from_page: 提示的翻页前页码（1 基）
    :param matched: 是否匹配到一次踏板操作；False 即漏翻
    :param pedal_event_index: 匹配到的踏板事件 0 基序号；漏翻为 None
    :param pedal_seconds: 匹配踏板事件的操作时刻（秒）；漏翻为 None
    :param time_diff_seconds: 匹配对的时间差绝对值（秒）；漏翻为 None
    """

    cue_index: int
    expected_seconds: float
    to_page: int
    from_page: int
    matched: bool
    pedal_event_index: Optional[int]
    pedal_seconds: Optional[float]
    time_diff_seconds: Optional[float]


@dataclass(frozen=True)
class PedalAuditResult:
    """一次踏板日志校对的完整结果。"""

    total_duration: float  # 按重复段展开后的演奏总时长（秒）
    tolerance_seconds: float  # 本次校对采用的时间容差（秒，非负）
    cue_audits: List[CueAudit]  # 每条应翻提示的校对结果（按提示顺序）
    extra_turns: List[int]  # 未匹配踏板事件的 0 基序号（多翻，升序）
    missed_count: int  # 漏翻数（未匹配提示数）
    extra_count: int  # 多翻数（未匹配踏板事件数）
    total_error_count: int  # 漏翻与多翻总数（第一项代价，全局最小）
    total_time_diff_seconds: float  # 匹配时间差之和（第二项代价，全局最小）


def _time_eps(magnitude: float) -> float:
    """演奏时间轴边界（总时长、非降序）比较用浮点容差，随数量级缩放。"""
    return 1e-9 * max(1.0, abs(magnitude))


def _match_eps(tolerance: float, diff: float) -> float:
    """单次匹配的容差边界比较用浮点容差，随容差与时间差的数量级缩放。"""
    return 1e-9 * max(1.0, tolerance, diff)


def audit_pedal_log(
    cue_seconds: Sequence[float],
    cue_to_pages: Sequence[int],
    cue_from_pages: Sequence[int],
    events: Sequence[PedalEvent],
    total_duration: float,
    page_count: int,
    tolerance_seconds: float,
) -> PedalAuditResult:
    """把按时间排列的踏板日志与应翻页提示做全局对齐校对。

    :param cue_seconds: 每条应翻提示的边界时刻（秒，按时间升序）
    :param cue_to_pages: 每条提示的目标页（1 基）；与 ``cue_seconds`` 等长
    :param cue_from_pages: 每条提示的翻页前页（1 基）；与 ``cue_seconds`` 等长，
        仅用于结果回显，不参与匹配
    :param events: 按时间**非降序**排列的踏板事件（秒数、操作后页码）
    :param total_duration: 按重复段展开后的演奏总时长（秒），事件秒数须落在
        ``[0, total_duration)``（恰等于总时长视为越出演奏时长）
    :param page_count: 已存方案页数，事件操作后页码须在 ``1..page_count``
    :param tolerance_seconds: 非负时间容差（秒，有限数值），仅当时间差不超过
        该容差且目标页相同时允许匹配
    :return: 每条提示的匹配/漏翻结果、多翻事件序号与两项代价
    :raises PedalAuditError: 任一输入约束不满足
    """
    m = len(cue_seconds)
    if not (
        len(cue_to_pages) == m and len(cue_from_pages) == m
    ):
        raise PedalAuditError("应翻提示的时刻、目标页与来源页数组长度必须一致")
    n = len(events)
    if n > MAX_EVENTS:
        raise PedalAuditError(
            f"踏板事件数（{n}）超过上限 {MAX_EVENTS}，无法校对"
        )
    if m > MAX_EVENTS:
        raise PedalAuditError(
            f"应翻页提示数（{m}）超过上限 {MAX_EVENTS}，无法校对"
        )

    if not isinstance(tolerance_seconds, (int, float)) or isinstance(
        tolerance_seconds, bool
    ):
        raise PedalAuditError("时间容差必须是数值")
    tol = float(tolerance_seconds)
    if tol != tol or tol in (float("inf"), float("-inf")):
        raise PedalAuditError("时间容差必须是有限数值（不接受 NaN 或 Infinity）")
    if tol < 0:
        raise PedalAuditError(f"时间容差不能为负数（实际 {tol:g} 秒）")

    if not isinstance(total_duration, (int, float)) or total_duration <= 0:
        raise PedalAuditError("演奏总时长必须为正数")
    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1:
        raise PedalAuditError("规划页数必须为正整数")

    checked: List[PedalEvent] = []
    prev_seconds: Optional[float] = None
    for idx, event in enumerate(events):
        if not isinstance(event, PedalEvent):
            raise PedalAuditError(f"第 {idx + 1} 个踏板事件格式不合法")
        seconds = event.seconds
        page = event.page_after
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise PedalAuditError(f"第 {idx + 1} 个踏板事件秒数必须是数值")
        seconds = float(seconds)
        if seconds != seconds or seconds in (float("inf"), float("-inf")):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件秒数必须是有限数值"
                "（不接受 NaN 或 Infinity）"
            )
        if seconds < 0:
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件：操作时刻 {seconds:g} 秒不能为负数"
                "（以演奏开始为 0）"
            )
        if seconds >= total_duration - _time_eps(total_duration):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件：操作时刻 {seconds:g} 秒越出演奏时长"
                f" {float(total_duration):g} 秒（须在 [0, 总时长) 内，"
                "恰等于总时长视为演奏结束之后）"
            )
        if not isinstance(page, int) or isinstance(page, bool):
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件的操作后页码必须为整数（收到 {page!r}）"
            )
        if not 1 <= page <= page_count:
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件：操作后页码 {page} 越出页码范围"
                f" 1..{page_count}"
            )
        if prev_seconds is not None and seconds + _time_eps(prev_seconds) < prev_seconds:
            raise PedalAuditError(
                f"第 {idx + 1} 个踏板事件：操作时刻 {seconds:g} 秒早于上一条"
                f" {prev_seconds:g} 秒，踏板事件必须按时间非降序排列"
            )
        checked.append(PedalEvent(seconds, page))
        prev_seconds = seconds

    if m == 0:
        # 整篇只有一页：没有任何应翻提示，所有踏板事件都是多翻
        return PedalAuditResult(
            total_duration=float(total_duration),
            tolerance_seconds=tol,
            cue_audits=[],
            extra_turns=list(range(n)),
            missed_count=0,
            extra_count=n,
            total_error_count=n,
            total_time_diff_seconds=0.0,
        )

    return _align(
        cue_seconds=[float(t) for t in cue_seconds],
        cue_to_pages=list(cue_to_pages),
        cue_from_pages=list(cue_from_pages),
        events=checked,
        total_duration=float(total_duration),
        tol=tol,
    )


def _align(
    cue_seconds: List[float],
    cue_to_pages: List[int],
    cue_from_pages: List[int],
    events: List[PedalEvent],
    total_duration: float,
    tol: float,
) -> PedalAuditResult:
    """后缀标量 DP + 记忆化字典序重构（两侧均已校验、非空提示）。"""
    m, n = len(cue_seconds), len(events)

    # compatible[i][j]：提示 i 与事件 j 目标页相同且时间差不超过容差；
    # weight[i][j] 为匹配对的时间差绝对值。
    compatible: List[bytearray] = [bytearray(n) for _ in range(m)]
    weight: List[List[float]] = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ct, cp = cue_seconds[i], cue_to_pages[i]
        row_c, row_w = compatible[i], weight[i]
        for j in range(n):
            if events[j].page_after == cp:
                diff = abs(events[j].seconds - ct)
                if diff <= tol + _match_eps(tol, diff):
                    row_c[j] = 1
                    row_w[j] = diff

    # 时间差之和一律用浮点的精确比较（< / ==）。每个 S 值都是由同一张 weight 表
    # 逐次相加得到，重构时用与填充完全相同的表达式（S[后继] + weight[i][j]）复算，
    # Python 浮点加法确定且可复现，故同一动作在填充与重构处必然逐位一致——既不会
    # 把真实不同（哪怕仅差 1e-15）的两项代价误判为并列，也不会让重构漏掉最优动作。
    def _same_cost(g: int, s: float, g_ref: int, s_ref: float) -> bool:
        return g == g_ref and s == s_ref

    # 后缀标量 DP：G = 最小（漏翻+多翻）数，S = 该数下的最小时间差之和。
    gaps = [[0] * (n + 1) for _ in range(m + 1)]
    sums = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        gaps[i][n] = m - i  # 事件耗尽：剩余提示全部漏翻
    for j in range(n):
        gaps[m][j] = n - j  # 提示耗尽：剩余事件全部多翻
    for i in range(m - 1, -1, -1):
        gi, gi_next = gaps[i], gaps[i + 1]
        si, si_next = sums[i], sums[i + 1]
        row_c, row_w = compatible[i], weight[i]
        for j in range(n - 1, -1, -1):
            # 候选 1：漏翻提示 i
            best_g, best_s = gi_next[j] + 1, si_next[j]
            # 候选 2：事件 j 多翻
            g2, s2 = gi[j + 1] + 1, si[j + 1]
            if g2 < best_g or (g2 == best_g and s2 < best_s):
                best_g, best_s = g2, s2
            # 候选 3：匹配
            if row_c[j]:
                g3, s3 = gi_next[j + 1], si_next[j + 1] + row_w[j]
                if g3 < best_g or (g3 == best_g and s3 < best_s):
                    best_g, best_s = g3, s3
            gi[j], si[j] = best_g, best_s

    # best(i, j)：后缀 [i:]/[j:] 在保持全局最优 (G, S) 前提下的字典序最小
    # 匹配事件序号元组。同一格至多计算一次；记忆化在下方 lru_cache 中完成。
    @lru_cache(maxsize=None)
    def _best(i: int, j: int) -> Tuple[int, ...]:
        if i == m or j == n:
            return ()
        g_ref, s_ref = gaps[i][j], sums[i][j]
        candidates: List[Tuple[int, ...]] = []
        # 三种动作中凡能保持最优代价者，收集其后续的最优元组后取字典序最小。
        if _same_cost(gaps[i + 1][j] + 1, sums[i + 1][j], g_ref, s_ref):
            candidates.append(_best(i + 1, j))
        if _same_cost(gaps[i][j + 1] + 1, sums[i][j + 1], g_ref, s_ref):
            candidates.append(_best(i, j + 1))
        if compatible[i][j] and _same_cost(
            gaps[i + 1][j + 1],
            sums[i + 1][j + 1] + weight[i][j],
            g_ref,
            s_ref,
        ):
            candidates.append((j,) + _best(i + 1, j + 1))
        return min(candidates) if candidates else ()

    # 前向重构：沿 _best 指示的最优路径逐格确定动作。
    matched_by_cue: List[Optional[int]] = [None] * m
    used_events: set[int] = set()
    i = j = 0
    while i < m or j < n:
        if i == m:
            j += 1  # 剩余事件均为多翻
            continue
        if j == n:
            i += 1  # 剩余提示均为漏翻
            continue
        g_ref, s_ref = gaps[i][j], sums[i][j]
        if compatible[i][j] and _same_cost(
            gaps[i + 1][j + 1],
            sums[i + 1][j + 1] + weight[i][j],
            g_ref,
            s_ref,
        ):
            match_tuple = (j,) + _best(i + 1, j + 1)
            if match_tuple == _best(i, j):
                matched_by_cue[i] = j
                used_events.add(j)
                i += 1
                j += 1
                continue
        if _same_cost(
            gaps[i + 1][j] + 1, sums[i + 1][j], g_ref, s_ref
        ) and _best(i + 1, j) == _best(i, j):
            i += 1  # 提示 i 漏翻
        else:
            j += 1  # 事件 j 多翻

    cue_audits: List[CueAudit] = []
    missed_count = 0
    for i in range(m):
        j = matched_by_cue[i]
        if j is None:
            missed_count += 1
            cue_audits.append(
                CueAudit(
                    cue_index=i,
                    expected_seconds=cue_seconds[i],
                    to_page=cue_to_pages[i],
                    from_page=cue_from_pages[i],
                    matched=False,
                    pedal_event_index=None,
                    pedal_seconds=None,
                    time_diff_seconds=None,
                )
            )
        else:
            cue_audits.append(
                CueAudit(
                    cue_index=i,
                    expected_seconds=cue_seconds[i],
                    to_page=cue_to_pages[i],
                    from_page=cue_from_pages[i],
                    matched=True,
                    pedal_event_index=j,
                    pedal_seconds=events[j].seconds,
                    time_diff_seconds=weight[i][j],
                )
            )

    extra_turns = [j for j in range(n) if j not in used_events]
    total_diff = sums[0][0]
    if total_diff == 0:
        total_diff = 0.0
    return PedalAuditResult(
        total_duration=total_duration,
        tolerance_seconds=tol,
        cue_audits=cue_audits,
        extra_turns=extra_turns,
        missed_count=missed_count,
        extra_count=len(extra_turns),
        total_error_count=gaps[0][0],
        total_time_diff_seconds=total_diff,
    )


# _best 的记忆化递归深度最多为 m + n（≤ 2·MAX_EVENTS），放宽默认上限。
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
