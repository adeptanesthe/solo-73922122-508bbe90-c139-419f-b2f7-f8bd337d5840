"""配对评估的独立对照验证。

用暴力枚举(枚举 A 的每条记录"跳过/与某条未用 B 配对")计算
  1) 最优 (配对数, 相似度之和) 与全部最优方案数;
  2) 最优方案中恢复率之差的最小值、最大值;
再与 app.pairing.evaluate 的 DP 结果逐项比对。随机生成大量小规模用例。

用法: python3 scripts/verify_pairing.py
退出码非 0 表示不一致。
"""
from __future__ import annotations

import itertools
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import build_state  # noqa: E402
from app.pairing import MIN_SIMILARITY, GroupItem, jaccard, evaluate  # noqa: E402


def brute(a_items, b_items, tokens):
    a_items = sorted(a_items, key=lambda x: x.id)
    b_items = sorted(b_items, key=lambda x: x.id)
    weights = {}
    for i, a in enumerate(a_items):
        for j, b in enumerate(b_items):
            sim = jaccard(tokens[a.id], tokens[b.id])
            if sim >= MIN_SIMILARITY:
                weights[(i, j)] = sim
    outcomes = []

    def rec(i, used, size, sim, score, pairs):
        if i == len(a_items):
            outcomes.append((size, sim, score, tuple(pairs)))
            return
        rec(i + 1, used, size, sim, score, pairs)
        for j in range(len(b_items)):
            if j in used or (i, j) not in weights:
                continue
            a = a_items[i]
            b = b_items[j]
            pairs.append((a.id, b.id, weights[(i, j)]))
            rec(
                i + 1,
                used | {j},
                size + 1,
                sim + weights[(i, j)],
                score + (1 if a.recovered else 0) - (1 if b.recovered else 0),
                pairs,
            )
            pairs.pop()

    rec(0, frozenset(), 0, Fraction(0), 0, [])
    best = max((o[0], o[1]) for o in outcomes)
    optimal = [o for o in outcomes if (o[0], o[1]) == best]
    return (
        best[0],
        best[1],
        len(optimal),
        min(o[2] for o in optimal),
        max(o[2] for o in optimal),
        {frozenset((p[0], p[1]) for p in o[3]) for o in optimal},
    )


def main() -> int:
    rng = random.Random(20261118)
    vocab = [f"w{i}" for i in range(7)]
    failures = 0
    trials = 400
    for trial in range(trials):
        na = rng.randint(1, 4)
        nb = rng.randint(1, 4)
        tokens = {}
        for i in range(na):
            tokens[f"A{i}"] = tuple(rng.choice(vocab) for _ in range(rng.randint(0, 3)))
        for j in range(nb):
            tokens[f"B{j}"] = tuple(rng.choice(vocab) for _ in range(rng.randint(0, 3)))
        state = build_state({k: ("", v) for k, v in tokens.items()})
        a_items = [GroupItem(f"A{i}", rng.random() < 0.5) for i in range(na)]
        b_items = [GroupItem(f"B{j}", rng.random() < 0.5) for j in range(nb)]
        ev = evaluate(state, a_items, b_items)
        size, sim, count, min_score, max_score, optimal_sets = brute(a_items, b_items, tokens)

        ok = True
        if (ev.paired, ev.similarity_sum, ev.optimal_pairing_count) != (size, sim, count):
            ok = False
        if size == 0:
            if ev.min_outcome is not None or ev.max_outcome is not None:
                ok = False
        else:
            if ev.min_outcome.score != min_score or ev.max_outcome.score != max_score:
                ok = False
            for outcome in (ev.min_outcome, ev.max_outcome):
                pair_set = frozenset((p.a_id, p.b_id) for p in outcome.pairs)
                if pair_set not in optimal_sets:
                    ok = False
                if len(outcome.pairs) != size:
                    ok = False
        if not ok:
            failures += 1
            print(f"[FAIL] trial {trial}: A={[(x.id, x.recovered) for x in a_items]} "
                  f"B={[(x.id, x.recovered) for x in b_items]}")
            print(f"  dp     : {ev.paired}, {ev.similarity_sum}, {ev.optimal_pairing_count}, "
                  f"{None if ev.min_outcome is None else ev.min_outcome.score}, "
                  f"{None if ev.max_outcome is None else ev.max_outcome.score}")
            print(f"  brute  : {size}, {sim}, {count}, {min_score}, {max_score}")

    print(f"{trials} 组随机用例与暴力枚举逐项比对完成,不一致 {failures} 组")

    # 手工示例:多最优、差值极值不同
    state = build_state({
        "A1": ("", ("x",)), "A2": ("", ("x",)), "A3": ("", ("z",)),
        "B1": ("", ("x", "y")),
    })
    ev = evaluate(
        state,
        [GroupItem("A1", True), GroupItem("A2", False), GroupItem("A3", True)],
        [GroupItem("B1", False)],
    )
    print("\n手工示例(A1 恢复/A2 未恢复/B1 未恢复,A1、A2 均与 B1 相似 1/2):")
    print(f"  最优配对 {ev.optimal_pairing_count} 种;配对数 {ev.paired};相似度和 {ev.similarity_sum}")
    print(f"  恢复率之差 最小 {ev.min_outcome.difference} ({[p.a_id for p in ev.min_outcome.pairs]})"
          f";最大 {ev.max_outcome.difference} ({[p.a_id for p in ev.max_outcome.pairs]})")
    if ev.min_outcome.difference == 0 and ev.max_outcome.difference == 1:
        print("  符合预期")
    else:
        print("  [FAIL] 与预期不符")
        failures += 1

    # 8×8 全同集合:最优方案数须为 8!
    spec = {f"A{i}": ("", ("a", "b")) for i in range(8)}
    spec.update({f"B{j}": ("", ("a", "b")) for j in range(8)})
    st8 = build_state(spec)
    ev8 = evaluate(
        st8,
        [GroupItem(f"A{i}", i % 2 == 0) for i in range(8)],
        [GroupItem(f"B{j}", j % 2 == 1) for j in range(8)],
    )
    expected = len(list(itertools.permutations(range(8))))
    print(f"\n8×8 全同集合:配对数 {ev8.paired},最优方案数 {ev8.optimal_pairing_count}(期望 {expected})")
    if ev8.optimal_pairing_count != expected:
        failures += 1

    print("\n结果:", "全部通过" if failures == 0 else f"失败 {failures} 处")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
