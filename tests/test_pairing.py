"""配对评估测试:相似度、阈值、目标优先级、多最优的差值极值、输入拒绝、版本一致。"""
import random
import threading
import time
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.index import IndexManager, build_state
from app.main import create_app
from app.pairing import (
    MIN_SIMILARITY,
    GroupItem,
    PairingInputError,
    evaluate,
    jaccard,
    validate_groups,
)


# ---------- 独立参照:暴力枚举所有配对方案 ----------

def brute(state, a_items, b_items):
    """枚举全部匹配,返回 (配对数, 相似度和, 最优方案数, 最小差值明细, 最大差值明细)。

    明细为 (score, pairs),pairs 为 [(a_id, b_id, Fraction)]。仅用于小规模验证。
    """
    a_items = sorted(a_items, key=lambda x: x.id)
    b_items = sorted(b_items, key=lambda x: x.id)
    weights = {}
    for i, a in enumerate(a_items):
        for j, b in enumerate(b_items):
            sim = jaccard(state.docs[a.id][1], state.docs[b.id][1])
            if sim >= MIN_SIMILARITY:
                weights[(i, j)] = sim

    all_matchings = []

    def rec(i, used, pairs):
        if i == len(a_items):
            size = len(pairs)
            sim = sum((w for _, _, w in pairs), Fraction(0))
            score = sum(
                (1 if a_items[ai].recovered else 0) - (1 if b_items[bj].recovered else 0)
                for ai, bj, _ in pairs
            )
            detail = (
                score,
                [(a_items[ai].id, b_items[bj].id, w) for ai, bj, w in pairs],
            )
            all_matchings.append(((size, sim), detail))
            return
        rec(i + 1, used, pairs)  # 跳过该 A
        for j in range(len(b_items)):
            if j in used or (i, j) not in weights:
                continue
            rec(i + 1, used | {j}, pairs + [(i, j, weights[(i, j)])])

    rec(0, frozenset(), [])
    best_key = max(k for k, _ in all_matchings)
    optimal = [d for k, d in all_matchings if k == best_key]
    optimal_pair_sets = {
        frozenset((a_id, b_id) for a_id, b_id, _ in d[1]) for d in optimal
    }
    return (
        best_key[0],
        best_key[1],
        len(optimal),
        min(optimal, key=lambda d: d[0]),
        max(optimal, key=lambda d: d[0]),
        optimal_pair_sets,
    )


def make_state(spec):
    """spec: {编号: [tokens]};正文留空,相似度只看 tokens。"""
    return build_state({k: ("", tuple(v)) for k, v in spec.items()})


# ---------- 相似度 ----------

def test_jaccard_dedupes_and_handles_empty():
    assert jaccard(["a", "a", "b"], ["a", "b"]) == 1
    assert jaccard(["a", "a", "a"], ["a", "b"]) == Fraction(1, 2)
    assert jaccard(["a", "b"], ["a", "b", "c", "d"]) == Fraction(1, 2)
    assert jaccard([], ["a"]) == 0
    assert jaccard([], []) == 0  # 空集合不构成配对
    # 词频不影响相似度
    assert jaccard(["a", "b"] * 5, ["a", "b"]) == 1


def test_threshold_is_inclusive():
    st = make_state({"A": ["a", "b"], "B": ["a", "b", "c", "d"]})  # 2/4 = 0.5
    ev = evaluate(st, [GroupItem("A", True)], [GroupItem("B", False)])
    assert ev.paired == 1
    st2 = make_state({"A": ["a", "b"], "B": ["a", "b", "c", "d", "e"]})  # 2/5 = 0.4
    ev2 = evaluate(st2, [GroupItem("A", True)], [GroupItem("B", False)])
    assert ev2.paired == 0 and ev2.min_outcome is None


# ---------- 目标优先级 ----------

def test_max_cardinality_beats_similarity_sum():
    # (A1,B1)=0.5 与 (A2,B2)=1 可同时成立 => 2 对;只取 (A2,B1)=1 则 1 对
    st = make_state({
        "A1": ["a", "b"],
        "A2": ["c", "d"],
        "B1": ["a", "b", "c", "d"],  # A1 相似 0.5;A2 相似 0.5
        "B2": ["c", "d"],            # A2 相似 1
    })
    ev = evaluate(
        st, [GroupItem("A1", True), GroupItem("A2", True)],
        [GroupItem("B1", False), GroupItem("B2", False)],
    )
    assert ev.paired == 2
    pairs = {(p.a_id, p.b_id): p.similarity for p in ev.min_outcome.pairs}
    assert pairs == {("A1", "B1"): Fraction(1, 2), ("A2", "B2"): Fraction(1, 1)}


def test_similarity_sum_is_second_priority():
    # A1={a} A2={a,b} B1={a,b} B2={a}
    # A1-B1=1/2, A1-B2=1, A2-B1=1, A2-B2=1/2
    # 两种 2 对方案:A1-B1+A2-B2=1,A1-B2+A2-B1=2;后者相似度和更大
    st = make_state({"A1": ["a"], "A2": ["a", "b"], "B1": ["a", "b"], "B2": ["a"]})
    ev = evaluate(
        st, [GroupItem("A1", True), GroupItem("A2", True)],
        [GroupItem("B1", False), GroupItem("B2", False)],
    )
    assert ev.paired == 2
    assert ev.similarity_sum == 2
    assert ev.optimal_pairing_count == 1
    assert {(p.a_id, p.b_id) for p in ev.min_outcome.pairs} == {("A1", "B2"), ("A2", "B1")}


# ---------- 多最优:必须同时给出差值极值与各自明细 ----------

def test_multiple_optimal_pairings_report_min_and_max_with_details():
    # 只有 1 条"可达阈值"的 B,A1、A2 都能与它配且相似度同为 1/2 => 2 种最优配对
    # A1 恢复、A2 未恢复、B1 未恢复:差值为 1 或 0,极值与明细须同时给出
    st = make_state({"A1": ["x"], "A2": ["x"], "A3": ["zzz"], "B1": ["x", "y"]})
    ev = evaluate(
        st,
        [GroupItem("A1", True), GroupItem("A2", False), GroupItem("A3", True)],
        [GroupItem("B1", False)],
    )
    assert ev.paired == 1
    assert ev.similarity_sum == Fraction(1, 2)
    assert ev.optimal_pairing_count == 2
    assert ev.unique_optimal_pairing is False
    assert ev.min_outcome.difference == Fraction(0)
    assert ev.max_outcome.difference == Fraction(1)
    assert [(p.a_id, p.b_id) for p in ev.min_outcome.pairs] == [("A2", "B1")]
    assert [(p.a_id, p.b_id) for p in ev.max_outcome.pairs] == [("A1", "B1")]


def test_unique_optimal_reports_single_difference():
    st = make_state({"A1": ["x", "y"], "A2": ["q"], "B1": ["x", "y"], "B2": ["q"]})
    ev = evaluate(
        st, [GroupItem("A1", True), GroupItem("A2", True)],
        [GroupItem("B1", False), GroupItem("B2", False)],
    )
    assert ev.unique_optimal_pairing is True
    assert ev.optimal_pairing_count == 1
    assert ev.min_outcome.difference == ev.max_outcome.difference == Fraction(1)
    assert [p.a_id for p in ev.min_outcome.pairs] == ["A1", "A2"]


def test_no_pairing_is_insufficient_evidence():
    st = make_state({"A1": ["x"], "B1": ["y"]})
    ev = evaluate(st, [GroupItem("A1", True)], [GroupItem("B1", True)])
    assert ev.paired == 0
    assert ev.optimal_pairing_count == 1  # 只有退化的空配对
    assert ev.min_outcome is None and ev.max_outcome is None
    assert ev.unique_optimal_pairing is False


def test_request_order_does_not_change_result():
    st = make_state({
        "A1": ["x"], "A2": ["x", "y"], "A3": ["p"],
        "B1": ["x"], "B2": ["x", "y"], "B3": ["p", "q"],
    })
    a = [GroupItem("A1", True), GroupItem("A2", False), GroupItem("A3", True)]
    b = [GroupItem("B1", False), GroupItem("B2", True), GroupItem("B3", True)]
    ev1 = evaluate(st, a, b)
    ev2 = evaluate(st, list(reversed(a)), list(reversed(b)))
    assert ev1.paired == ev2.paired
    assert ev1.similarity_sum == ev2.similarity_sum
    assert ev1.optimal_pairing_count == ev2.optimal_pairing_count
    assert ev1.min_outcome.difference == ev2.min_outcome.difference
    assert ev1.max_outcome.difference == ev2.max_outcome.difference
    pairs1 = {(p.a_id, p.b_id, p.similarity) for p in ev1.min_outcome.pairs}
    pairs2 = {(p.a_id, p.b_id, p.similarity) for p in ev2.min_outcome.pairs}
    assert pairs1 == pairs2


# ---------- 随机对照暴力枚举 ----------

def test_random_against_bruteforce():
    rng = random.Random(20261118)
    vocab = ["a", "b", "c", "d", "e", "f"]
    for trial in range(150):
        na, nb = rng.randint(1, 4), rng.randint(1, 4)
        spec = {}
        for i in range(na):
            spec[f"A{i}"] = [rng.choice(vocab) for _ in range(rng.randint(0, 4))]
        for j in range(nb):
            spec[f"B{j}"] = [rng.choice(vocab) for _ in range(rng.randint(0, 4))]
        st = make_state(spec)
        a_items = [GroupItem(f"A{i}", rng.random() < 0.5) for i in range(na)]
        b_items = [GroupItem(f"B{j}", rng.random() < 0.5) for j in range(nb)]
        ev = evaluate(st, a_items, b_items)
        size, sim, count, min_detail, max_detail, optimal_sets = brute(st, a_items, b_items)
        assert (ev.paired, ev.similarity_sum, ev.optimal_pairing_count) == (size, sim, count), spec
        if size == 0:
            assert ev.min_outcome is None and ev.max_outcome is None
            continue
        assert ev.optimal_pairing_count >= 1
        assert ev.min_outcome.score == min_detail[0]
        assert ev.max_outcome.score == max_detail[0]
        # 同差值可能对应多个方案;参照按差值校验,明细只校验其为合法最优配对
        for outcome, detail_score in (
            (ev.min_outcome, min_detail[0]),
            (ev.max_outcome, max_detail[0]),
        ):
            assert (outcome.paired, outcome.similarity_sum) == (size, sim)
            assert outcome.score == detail_score
            assert len(outcome.pairs) == size
            a_ids = [p.a_id for p in outcome.pairs]
            b_ids = [p.b_id for p in outcome.pairs]
            assert len(set(a_ids)) == len(a_ids)
            assert len(set(b_ids)) == len(b_ids)
            # 给出的明细必须是枚举中真实存在的某种最优配对
            assert frozenset((p.a_id, p.b_id) for p in outcome.pairs) in optimal_sets
            for p in outcome.pairs:
                assert p.similarity >= MIN_SIMILARITY


def test_eight_by_eight_full_match():
    # 8×8 全同集合:配对数 8,方案数 = 8! = 40320(全部达到最优)
    spec = {"A%d" % i: ["a", "b"] for i in range(8)}
    spec.update({"B%d" % j: ["a", "b"] for j in range(8)})
    st = make_state(spec)
    a = [GroupItem(f"A{i}", i % 2 == 0) for i in range(8)]
    b = [GroupItem(f"B{j}", j % 2 == 1) for j in range(8)]
    ev = evaluate(st, a, b)
    assert ev.paired == 8
    assert ev.optimal_pairing_count == 40320
    assert ev.min_outcome.difference == 0
    assert ev.max_outcome.difference == 0
    assert len(ev.min_outcome.pairs) == 8


# ---------- 编号校验 ----------

def test_validate_rejects_missing_and_duplicates():
    st = make_state({"A1": ["x"], "B1": ["x"]})
    with pytest.raises(PairingInputError) as exc:
        validate_groups(
            st,
            [GroupItem("A1", True), GroupItem("A1", True)],
            [GroupItem("NOPE", False)],
        )
    assert exc.value.duplicates == ["A1"]
    assert exc.value.missing == ["NOPE"]
    # 跨组重复也算重复
    with pytest.raises(PairingInputError) as exc:
        validate_groups(st, [GroupItem("A1", True)], [GroupItem("A1", True)])
    assert exc.value.duplicates == ["A1"]


# ---------- API ----------

@pytest.fixture()
def client():
    return TestClient(create_app(IndexManager()))


def upsert(client, docs):
    r = client.post(
        "/documents/batch",
        json={"upserts": [{"id": i, "text": "", "tokens": t} for i, t in docs.items()]},
    )
    assert r.status_code == 200
    return r.json()


def test_api_insufficient_evidence(client):
    upsert(client, {"A1": ["x"], "B1": ["y"]})
    r = client.post(
        "/pairing/evaluate",
        json={"a": [{"id": "A1", "recovered": True}], "b": [{"id": "B1", "recovered": True}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["insufficient_evidence"] is True
    assert body["paired"] == 0
    assert body["min_difference"] is None and body["max_difference"] is None
    assert body["unique_optimal_pairing"] is False


def test_api_multiple_optima_exposes_both(client):
    upsert(client, {"A1": ["x"], "A2": ["x"], "A3": ["zzz"], "B1": ["x", "y"]})
    r = client.post(
        "/pairing/evaluate",
        json={
            "a": [
                {"id": "A1", "recovered": True},
                {"id": "A2", "recovered": False},
                {"id": "A3", "recovered": True},
            ],
            "b": [{"id": "B1", "recovered": False}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["optimal_pairing_count"] == 2
    assert body["unique_optimal_pairing"] is False
    assert Fraction(body["min_difference"]["difference"]) == Fraction(0)
    assert Fraction(body["max_difference"]["difference"]) == Fraction(1)
    assert body["min_difference"]["pairs"][0]["a_id"] == "A2"
    assert body["max_difference"]["pairs"][0]["a_id"] == "A1"
    assert "不唯一" in body["message"]


def test_api_rejects_whole_request_on_missing_or_duplicate(client):
    upsert(client, {"A1": ["x"], "B1": ["x"]})
    cases = [
        {"a": [{"id": "A1", "recovered": True}, {"id": "A1", "recovered": False}],
         "b": [{"id": "B1", "recovered": True}]},
        {"a": [{"id": "A1", "recovered": True}],
         "b": [{"id": "MISSING", "recovered": True}]},
        {"a": [{"id": "A1", "recovered": True}],
         "b": [{"id": "A1", "recovered": True}]},
    ]
    for payload in cases:
        r = client.post("/pairing/evaluate", json=payload)
        assert r.status_code == 422, payload
        assert r.json()["detail"]["message"] == "请求被整体拒绝,未做任何评估"


def test_api_group_size_bounds(client):
    upsert(client, {f"D{i}": ["x"] for i in range(20)})
    ok = lambda n, m: client.post(
        "/pairing/evaluate",
        json={
            "a": [{"id": f"D{i}", "recovered": True} for i in range(n)],
            "b": [{"id": f"D{10 + j}", "recovered": False} for j in range(m)],
        },
    ).status_code
    assert ok(1, 1) == 200
    assert ok(8, 8) == 200
    assert ok(0, 1) == 422
    assert ok(1, 0) == 422
    assert ok(9, 1) == 422
    assert ok(1, 9) == 422


def test_api_rejects_non_boolean_recovery(client):
    upsert(client, {"A1": ["x"], "B1": ["x"]})
    for value in ("yes", "true", 1, None):
        r = client.post(
            "/pairing/evaluate",
            json={"a": [{"id": "A1", "recovered": value}],
                  "b": [{"id": "B1", "recovered": True}]},
        )
        assert r.status_code == 422, value


def test_api_empty_token_records_are_not_paired(client):
    upsert(client, {"E1": [], "E2": []})
    r = client.post(
        "/pairing/evaluate",
        json={"a": [{"id": "E1", "recovered": True}],
              "b": [{"id": "E2", "recovered": True}]},
    ).json()
    assert r["insufficient_evidence"] is True and r["paired"] == 0


def test_api_successful_correction_recomputes(client):
    upsert(client, {"A1": ["x", "y"], "B1": ["x", "y"]})
    payload = {
        "a": [{"id": "A1", "recovered": True}],
        "b": [{"id": "B1", "recovered": False}],
    }
    first = client.post("/pairing/evaluate", json=payload).json()
    assert first["paired"] == 1 and Fraction(first["min_difference"]["difference"]) == 1
    v1 = first["version"]

    # 成功更正:B1 分词改为不相交 => 新请求重算,无配对
    upsert(client, {"B1": ["完全不同"]})
    second = client.post("/pairing/evaluate", json=payload).json()
    assert second["version"] > v1
    assert second["insufficient_evidence"] is True
    assert second["paired"] == 0


def test_api_failed_correction_keeps_original_state(client):
    from app.index import MAX_DOCS

    upsert(client, {"A1": ["x", "y"], "B1": ["x", "y"]})
    payload = {
        "a": [{"id": "A1", "recovered": True}],
        "b": [{"id": "B1", "recovered": False}],
    }
    before = client.post("/pairing/evaluate", json=payload).json()

    big = [{"id": f"X{i}", "text": "", "tokens": ["x"]} for i in range(MAX_DOCS)]
    r = client.post("/documents/batch", json={"upserts": big})
    assert r.status_code == 409

    after = client.post("/pairing/evaluate", json=payload).json()
    assert after["version"] == before["version"]
    assert after["paired"] == before["paired"] == 1
    assert after["min_difference"] == before["min_difference"]


def test_api_pairs_and_conclusion_share_one_snapshot_during_slow_build(client):
    upsert(client, {"A1": ["x", "y"], "B1": ["x", "y"]})
    # 用同一 manager 开启慢建索引,构建期间发起配对请求:必须整体使用旧版本
    mgr = client.app.state.manager
    mgr._build_delay = 0.8
    results = {}

    def slow_update():
        mgr.apply_batch([("B1", "", ("完全不同",))], [])

    t = threading.Thread(target=slow_update)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.1)
    r = client.post(
        "/pairing/evaluate",
        json={
            "a": [{"id": "A1", "recovered": True}],
            "b": [{"id": "B1", "recovered": False}],
        },
    )
    results["elapsed"] = time.monotonic() - t0
    results["body"] = r.json()
    t.join()
    mgr._build_delay = 0.0

    assert results["elapsed"] < 0.6  # 未被慢建索引挡住
    body = results["body"]
    assert body["paired"] == 1  # 旧版本里两条记录仍相似
    assert body["version"] == 1

    # 构建完成后新请求使用新版本
    body2 = client.post(
        "/pairing/evaluate",
        json={
            "a": [{"id": "A1", "recovered": True}],
            "b": [{"id": "B1", "recovered": False}],
        },
    ).json()
    assert body2["version"] == 2 and body2["insufficient_evidence"] is True


def test_api_docs_and_openapi_available(client):
    docs = client.get("/docs")
    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    schema = client.get("/openapi.json").json()
    assert "/pairing/evaluate" in schema["paths"]
    assert "/search" in schema["paths"]  # 现有检索仍在
