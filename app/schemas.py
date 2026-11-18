"""请求/响应模型。"""
from __future__ import annotations

from fractions import Fraction

from pydantic import BaseModel, Field


class DocUpsert(BaseModel):
    """单篇记录的增改:编号、正文、分词序列。"""

    id: str = Field(min_length=1, max_length=128, description="记录编号")
    text: str = Field(default="", description="记录正文")
    tokens: list[str] = Field(default_factory=list, description="分词序列(可为空,空文档计入统计)")


class BatchRequest(BaseModel):
    """批量增改删:同一批内先删后增改。"""

    upserts: list[DocUpsert] = Field(default_factory=list, description="新增/修改的记录")
    deletes: list[str] = Field(default_factory=list, description="删除的编号")


class BatchResponse(BaseModel):
    status: str = "ok"
    total: int = Field(description="生效后的总篇数")
    upserted: int = Field(description="本批提交的增改条数")
    deleted: int = Field(description="本批实际删除的篇数")


class SearchRequest(BaseModel):
    terms: list[str] = Field(default_factory=list, description="查询词项(服务端再去重)")
    k: int = Field(ge=1, le=20, description="返回篇数,1~20")


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    scored: int = Field(description="完整计分篇数")


# ---- 配对评估 ----


class RepairRecord(BaseModel):
    """一条待评估的维修记录:现存记录编号 + 逐单人工核实的恢复结果。"""

    id: str = Field(min_length=1, max_length=128, description="现存维修记录编号")
    recovered: bool = Field(
        strict=True,
        description="人工核实的恢复结果,只接受 JSON 布尔值 true/false,不猜测其他写法",
    )


class PairingRequest(BaseModel):
    """A、B 两组各 1~8 条记录;编号须存在且不得重复。"""

    a: list[RepairRecord] = Field(min_length=1, max_length=8, description="A 组(措施 A)")
    b: list[RepairRecord] = Field(min_length=1, max_length=8, description="B 组(措施 B)")


class PairDetail(BaseModel):
    a_id: str
    b_id: str
    similarity: Fraction = Field(description="去重分词集合的 Jaccard 相似度(精确值)")
    similarity_float: float


class OutcomeDetail(BaseModel):
    """一种最优配对及其恢复率之差。"""

    difference: Fraction = Field(description="A 恢复率 - B 恢复率(精确值)")
    difference_float: float
    a_recovery_rate: Fraction
    b_recovery_rate: Fraction
    a_recovered: int
    b_recovered: int
    pairs: list[PairDetail]


class PairingResponse(BaseModel):
    version: int = Field(description="本次评估所用记录版本;同一请求的配对与结论均来自该版本")
    paired: int = Field(description="最优配对的配对数")
    similarity_sum: Fraction = Field(description="最优配对的相似度之和(精确值)")
    optimal_pairing_count: int = Field(
        description="达到最优目标的不同配对方案数;证据不足时为 1(退化的空配对)"
    )
    insufficient_evidence: bool = Field(description="无允许的跨组配对时为 true,此时差值为空")
    unique_optimal_pairing: bool = Field(description="最优配对是否唯一;不唯一时 min/max 均须参考")
    min_difference: OutcomeDetail | None = Field(
        default=None, description="最优配对中恢复率之差的最小值及其明细"
    )
    max_difference: OutcomeDetail | None = Field(
        default=None, description="最优配对中恢复率之差的最大值及其明细"
    )
    message: str
