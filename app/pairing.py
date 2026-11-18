"""同型号设备两种维修措施(A/B)的配对评估。

规则:
- 相似度 = 两组记录"去重后分词集合"的 Jaccard 系数;空集合视为相似度 0。
- 只允许相似度 >= 1/2 的跨组配对;每条记录至多出现在一个配对中。
- 目标按字典序:先使配对数最多,再使相似度之和最大。
- 在所有达到该目标的最优配对中,统计"恢复率之差"的最小值与最大值并各给出一份明细。
  恢复率 = 该组配对记录中恢复数 / 配对数(两组分母同为配对数)。
  差值 = A 恢复率 - B 恢复率 = (A 恢复数 - B 恢复数) / 配对数。

全部用精确有理数计算:阈值与相似度之和的比较不受浮点误差影响,
  DP 内部以整数标度(key 为公共分母 L 下的整数),仅在输出时还原为 Fraction。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .index import State

MIN_SIMILARITY = Fraction(1, 2)
# 一条方案在 DP 中的"恢复统计":(A 恢复数 - B 恢复数, A 恢复数, 配对明细)
_ZERO_RECORD = (0, 0, ())


def jaccard(tokens_a, tokens_b) -> Fraction:
    """去重分词集合的 Jaccard 相似度;两个集合都为空时返回 0。"""
    a = set(tokens_a)
    b = set(tokens_b)
    union = a | b
    if not union:
        return Fraction(0)
    return Fraction(len(a & b), len(union))


@dataclass(frozen=True)
class GroupItem:
    id: str
    recovered: bool


@dataclass(frozen=True)
class PairChoice:
    a_id: str
    b_id: str
    similarity: Fraction


@dataclass(frozen=True)
class Outcome:
    """一种完整配对及其恢复情况。"""

    paired: int
    similarity_sum: Fraction
    score: int  # A 恢复数 - B 恢复数
    a_recovered: int
    b_recovered: int
    pairs: tuple[PairChoice, ...]

    @property
    def difference(self) -> Fraction:
        return Fraction(self.score, self.paired)

    @property
    def a_rate(self) -> Fraction:
        return Fraction(self.a_recovered, self.paired)

    @property
    def b_rate(self) -> Fraction:
        return Fraction(self.b_recovered, self.paired)


@dataclass(frozen=True)
class Evaluation:
    """一次评估的结果。

    paired == 0 表示不存在允许的跨组配对(证据不足),此时 min/max 均为 None,
    optimal_pairing_count 为 1(只有"一个都不配"这一种退化方案),唯一性标记为 False。
    否则 min_outcome / max_outcome 分别给出"恢复率之差"最小、最大的最优配对的完整明细;
    两者可能相同(唯一最优配对)。
    """

    version: int
    paired: int
    similarity_sum: Fraction
    optimal_pairing_count: int
    min_outcome: Outcome | None
    max_outcome: Outcome | None

    @property
    def unique_optimal_pairing(self) -> bool:
        return self.paired > 0 and self.optimal_pairing_count == 1


def _min_record(cand, cur):
    """恢复统计的最小值比较:先比差值,再比 A 恢复数,最后按配对编号字典序。"""
    if cur is None:
        return True
    if cand[0] != cur[0]:
        return cand[0] < cur[0]
    if cand[1] != cur[1]:
        return cand[1] < cur[1]
    return cand[2] < cur[2]


def _max_record(cand, cur):
    """恢复统计的最大值比较:先比差值,再比 A 恢复数,最后按配对编号字典序。"""
    if cur is None:
        return True
    if cand[0] != cur[0]:
        return cand[0] > cur[0]
    if cand[1] != cur[1]:
        return cand[1] > cur[1]
    return cand[2] < cur[2]


def _weights(state: State, a_items: list[GroupItem], b_items: list[GroupItem]):
    """(i, j) -> 相似度(仅保留达到阈值的跨组配对)。"""
    weights: dict = {}
    for i, a in enumerate(a_items):
        for j, b in enumerate(b_items):
            sim = jaccard(state.docs[a.id][1], state.docs[b.id][1])
            if sim >= MIN_SIMILARITY:
                weights[(i, j)] = sim
    return weights


class PairingInputError(Exception):
    """请求中的编号缺失或重复:整次请求拒绝,不做任何评估。"""

    def __init__(self, missing: list[str], duplicates: list[str]):
        self.missing = missing
        self.duplicates = duplicates
        parts = []
        if missing:
            parts.append("编号不存在:" + "、".join(missing))
        if duplicates:
            parts.append("编号重复:" + "、".join(duplicates))
        super().__init__(";".join(parts))


def validate_groups(
    state: State, a_items: list[GroupItem], b_items: list[GroupItem]
) -> None:
    """在给定 State 上校验编号:必须存在,且 A、B 合并后每个编号只能出现一次。

    校验与评估必须使用同一个 State 快照:调用方先取 state,再依次调用
    validate_groups 与 evaluate,两者不会看到不同版本的记录。
    """
    combined = [x.id for x in a_items] + [x.id for x in b_items]
    seen: dict = {}
    duplicates: list = []
    for did in combined:
        seen[did] = seen.get(did, 0) + 1
        if seen[did] == 2:
            duplicates.append(did)
    missing = [did for did in seen if did not in state.docs]
    if missing or duplicates:
        raise PairingInputError(sorted(missing), sorted(duplicates))


def evaluate(state: State, a_items: list[GroupItem], b_items: list[GroupItem]) -> Evaluation:
    """在给定不可变 State 上完成一次配对评估(不读取任何外部状态)。

    调用方需先确认所有编号都存在于同一 State;这里按编号升序处理,
    使结果与请求内 A/B 两组的书写顺序无关。

    采用按 A 记录逐条推进的位掩码 DP(每组至多 8 条,B 集合至多 256 种):
    - G[i][mask]:从"已处理前 i 条 A、已占用 B 集合 mask"出发还能取得的最优
      (配对数, 相似度之和),字典序最大;状态数 9×256,规模固定。
    - 计数 / 最小 / 最大:只沿"能达到 G 的最优转移"推进。
      每条配对方案对应唯一一条决策路径(A_i 与哪条 B 配对或跳过),因此计数不重不漏。
    """
    a_items = sorted(a_items, key=lambda x: x.id)
    b_items = sorted(b_items, key=lambda x: x.id)
    na, nb = len(a_items), len(b_items)
    full_b = 1 << nb
    weights = _weights(state, a_items, b_items)

    # 相似度之和的精确整数标度:所有 Jaccard 值都以 L 为分母,
    # 因而 "相似度之和 * L" 恒为整数,DP 全程只做整数运算(避免每状态 Fraction 开销)。
    lcm = 1
    for w in weights.values():
        lcm = lcm * w.denominator // math.gcd(lcm, w.denominator)
    scaled = {k: (w.numerator * (lcm // w.denominator)) for k, w in weights.items()}
    zero_value = (0, 0)

    # 预计算:每条 A 在每个 B 占用集合下的全部转移(状态数固定为 9×256,一次就够)
    # 转移为 (next_mask, added_value, added_record);added_record 里的第三项是
    # 规范化配对键,逐条追加后天然按 A 编号升序,可直接用于确定性的同分比较。
    trans: list = []
    for i in range(na):
        a = a_items[i]
        a_inc = 1 if a.recovered else 0
        per_mask = []
        for mask in range(full_b):
            row = [(mask, zero_value, _ZERO_RECORD)]  # 跳过第 i 条 A
            for j in range(nb):
                w = weights.get((i, j))
                if w is None or (mask >> j) & 1:
                    continue
                b = b_items[j]
                delta = a_inc - (1 if b.recovered else 0)
                row.append(
                    (
                        mask | (1 << j),
                        (1, scaled[(i, j)]),
                        (delta, a_inc, ((a.id, b.id),)),
                    )
                )
            per_mask.append(row)
        trans.append(per_mask)

    # ---- 后缀最优值 G ----
    G = [[zero_value] * full_b for _ in range(na + 1)]
    for i in range(na - 1, -1, -1):
        nxt = G[i + 1]
        cur = G[i]
        per_mask = trans[i]
        for mask in range(full_b):
            best = nxt[mask]
            for nm, added, _ in per_mask[mask]:
                cand = (nxt[nm][0] + added[0], nxt[nm][1] + added[1])
                if cand > best:
                    best = cand
            cur[mask] = best

    opt = G[0][0]
    if opt[0] == 0:
        # 没有任何相似度达到阈值的跨组配对,只有退化的空配对
        return Evaluation(
            version=state.version,
            paired=0,
            similarity_sum=Fraction(0),
            optimal_pairing_count=1,
            min_outcome=None,
            max_outcome=None,
        )

    # ---- 只沿最优转移推进:计数、差值最小值、差值最大值 ----
    count = [[0] * full_b for _ in range(na + 1)]
    mins: list = [[None] * full_b for _ in range(na + 1)]
    maxs: list = [[None] * full_b for _ in range(na + 1)]
    for mask in range(full_b):
        count[na][mask] = 1
        mins[na][mask] = _ZERO_RECORD
        maxs[na][mask] = _ZERO_RECORD

    for i in range(na - 1, -1, -1):
        nxt_g = G[i + 1]
        nxt_count = count[i + 1]
        nxt_min = mins[i + 1]
        nxt_max = maxs[i + 1]
        cur_g = G[i]
        cur_count = count[i]
        cur_min = mins[i]
        cur_max = maxs[i]
        per_mask = trans[i]
        for mask in range(full_b):
            target = cur_g[mask]
            total = 0
            best_min = None
            best_max = None
            for nm, added, rec in per_mask[mask]:
                suffix = nxt_g[nm]
                if (suffix[0] + added[0], suffix[1] + added[1]) != target:
                    continue  # 该转移无法达到该状态的最优值
                total += nxt_count[nm]
                smin = nxt_min[nm]
                cand = (rec[0] + smin[0], rec[1] + smin[1], rec[2] + smin[2])
                if _min_record(cand, best_min):
                    best_min = cand
                smax = nxt_max[nm]
                cand = (rec[0] + smax[0], rec[1] + smax[1], rec[2] + smax[2])
                if _max_record(cand, best_max):
                    best_max = cand
            cur_count[mask] = total
            cur_min[mask] = best_min
            cur_max[mask] = best_max

    def to_outcome(record) -> Outcome:
        score, a_recovered, keys = record
        return Outcome(
            paired=opt[0],
            similarity_sum=Fraction(opt[1], lcm),
            score=score,
            a_recovered=a_recovered,
            b_recovered=a_recovered - score,
            pairs=tuple(
                PairChoice(a_id, b_id, weights[(index[a_id], index_b[b_id])])
                for a_id, b_id in keys
            ),
        )

    index = {item.id: i for i, item in enumerate(a_items)}
    index_b = {item.id: j for j, item in enumerate(b_items)}
    return Evaluation(
        version=state.version,
        paired=opt[0],
        similarity_sum=Fraction(opt[1], lcm),
        optimal_pairing_count=count[0][0],
        min_outcome=to_outcome(mins[0][0]),
        max_outcome=to_outcome(maxs[0][0]),
    )
