"""FastAPI 入口:设备维修记录 BM25 精确检索。

- POST /documents/batch      批量增改删(语料、全库统计、剪枝依据一起生效)
- POST /search               正式检索(BlockMax WAND 块上界剪枝)
- POST /search/exhaustive    对照接口(全量计分后排序,仅用于验证一致性)
- POST /pairing/evaluate     两种维修措施 A/B 的配对评估
- GET  /stats                当前状态统计
- GET  /healthz              健康检查
- GET  /docs                 Swagger UI
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .index import B, K1, MAX_DOCS, CapacityError, IndexManager, exhaustive_search, search
from .pairing import GroupItem, PairingInputError, evaluate as evaluate_pairing, validate_groups
from .schemas import (
    BatchRequest,
    BatchResponse,
    Hit,
    OutcomeDetail,
    PairDetail,
    PairingRequest,
    PairingResponse,
    SearchRequest,
    SearchResponse,
)


def create_app(manager: IndexManager | None = None) -> FastAPI:
    if manager is None:
        delay = float(os.environ.get("BM25_BUILD_DELAY", "0") or 0)
        manager = IndexManager(build_delay=delay)

    app = FastAPI(
        title="设备维修记录 BM25 检索",
        version="1.0.0",
        description=(
            "精确 BM25(k1=1.2, b=0.75, IDF=ln(1+(N-df+0.5)/(df+0.5)))。"
            "正式检索使用 BlockMax WAND 块上界剪枝,前 k 与穷举一致,同分按编号升序。"
        ),
    )
    app.state.manager = manager

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/healthz", tags=["meta"])
    def healthz():
        return {"status": "ok"}

    @app.get("/stats", tags=["meta"])
    def stats():
        s = manager.current()
        return {
            "docs": s.n,
            "avg_len": s.avgdl,
            "terms": len(s.terms),
            "version": s.version,
            "k1": K1,
            "b": B,
            "max_docs": MAX_DOCS,
        }

    @app.post("/documents/batch", response_model=BatchResponse, tags=["corpus"])
    def batch_update(req: BatchRequest):
        """批量增改删。语料、全库统计与剪枝依据一起生效;失败保留旧状态。"""
        try:
            new_state, deleted = manager.apply_batch(
                [(d.id, d.text, d.tokens) for d in req.upserts],
                list(req.deletes),
            )
        except CapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return BatchResponse(total=new_state.n, upserted=len(req.upserts), deleted=deleted)

    @app.post("/search", response_model=SearchResponse, tags=["search"])
    def search_ep(req: SearchRequest):
        """正式检索:块上界剪枝,返回命中编号、正文、分数及完整计分篇数。"""
        state = manager.current()  # 一次查询只用同一完整状态
        hits, scored = search(state, req.terms, req.k)
        return SearchResponse(
            hits=[Hit(id=i, text=t, score=sc) for i, t, sc in hits], scored=scored
        )

    @app.post("/search/exhaustive", response_model=SearchResponse, tags=["debug"])
    def search_exhaustive_ep(req: SearchRequest):
        """对照接口:全量计分后排序,仅用于验证正式检索的一致性,非正式路径。"""
        state = manager.current()
        hits, scored = exhaustive_search(state, req.terms, req.k)
        return SearchResponse(
            hits=[Hit(id=i, text=t, score=sc) for i, t, sc in hits], scored=scored
        )

    @app.post("/pairing/evaluate", response_model=PairingResponse, tags=["pairing"])
    def pairing_evaluate_ep(req: PairingRequest):
        """A/B 两组维修措施的配对评估。

        规则:去重分词集合的 Jaccard 相似度 >= 0.5 才允许跨组配对,每单至多用一次;
        先使配对数最多,再使相似度之和最大。在全部最优配对中给出恢复率之差的
        最小值和最大值及各自明细(最优唯一时两者相同);无配对时报证据不足。
        编号缺失或重复时整次拒绝(422)。
        校验、配对与统计都在同一个 State 快照上完成,对应同一版记录。
        """
        state = manager.current()  # 一次性取快照:校验与评估共用同一版本
        a_items = [GroupItem(r.id, r.recovered) for r in req.a]
        b_items = [GroupItem(r.id, r.recovered) for r in req.b]
        try:
            validate_groups(state, a_items, b_items)
        except PairingInputError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "请求被整体拒绝,未做任何评估",
                    "missing_ids": exc.missing,
                    "duplicate_ids": exc.duplicates,
                },
            ) from exc

        ev = evaluate_pairing(state, a_items, b_items)

        def outcome_detail(outcome) -> OutcomeDetail:
            return OutcomeDetail(
                difference=outcome.difference,
                difference_float=float(outcome.difference),
                a_recovery_rate=outcome.a_rate,
                b_recovery_rate=outcome.b_rate,
                a_recovered=outcome.a_recovered,
                b_recovered=outcome.b_recovered,
                pairs=[
                    PairDetail(
                        a_id=p.a_id,
                        b_id=p.b_id,
                        similarity=p.similarity,
                        similarity_float=float(p.similarity),
                    )
                    for p in outcome.pairs
                ],
            )

        if ev.paired == 0:
            return PairingResponse(
                version=ev.version,
                paired=0,
                similarity_sum=ev.similarity_sum,
                optimal_pairing_count=ev.optimal_pairing_count,
                insufficient_evidence=True,
                unique_optimal_pairing=False,
                min_difference=None,
                max_difference=None,
                message="证据不足:没有相似度达到 0.5 的跨组可配对记录",
            )

        min_detail = outcome_detail(ev.min_outcome)
        max_detail = outcome_detail(ev.max_outcome)
        if ev.unique_optimal_pairing:
            message = (
                f"最优配对唯一:共 {ev.paired} 对,A、B 恢复率之差为 "
                f"{ev.min_outcome.difference}"
            )
        else:
            message = (
                f"最优配对不唯一(共 {ev.optimal_pairing_count} 种):恢复率之差"
                f"最小 {ev.min_outcome.difference}、最大 {ev.max_outcome.difference},"
                "须同时参考两者的配对明细"
            )
        return PairingResponse(
            version=ev.version,
            paired=ev.paired,
            similarity_sum=ev.similarity_sum,
            optimal_pairing_count=ev.optimal_pairing_count,
            insufficient_evidence=False,
            unique_optimal_pairing=ev.unique_optimal_pairing,
            min_difference=min_detail,
            max_difference=max_detail,
            message=message,
        )

    return app


app = create_app()
