# 设备维修记录 BM25 精确检索

面向设备维护工程师的故障处理办法检索服务:FastAPI + NumPy 实现精确 BM25,
正式检索使用 BlockMax WAND 块上界剪枝(不全量计分后排序),前 k 与穷举完全一致。

另提供同型号设备两种维修措施(A/B)的配对评估接口,见下方
[配对评估](#post-pairingevaluate--ab-配对评估)。

## 快速启动

```bash
docker compose up --build
```

- 交互式 API 文档: <http://localhost:8000/docs>
- 健康检查: <http://localhost:8000/healthz>

载入本地示例语料(1500 篇维修记录,含空文档):

```bash
python3 scripts/load_corpus.py            # 默认 http://localhost:8000
```

本地开发(不用 Docker):

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
```

## 检索模型

- 固定 `k1 = 1.2`、`b = 0.75`;`IDF = ln(1 + (N − df + 0.5) / (df + 0.5))`(恒为正)。
- 文档得分 = Σ(去重查询词项) `idf · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`。
- 空文档(分词序列为空)计入 N 与平均长度 avgdl,不产生任何 posting,永远不会成为命中。
- 空库检索返回空结果(`{"hits": [], "scored": 0}`)。
- 同分按编号升序;查询词项服务端去重;`k` 限 1~20。

## 接口

### `POST /documents/batch` — 批量增改删

```json
{
  "upserts": [{"id": "R0001", "text": "数控机床轴承过热,更换轴承后恢复正常。", "tokens": ["数控机床", "轴承过热", "更换轴承"]}],
  "deletes": ["R0002"]
}
```

- 同一批内先删后增改;增改中重复编号后者覆盖前者。
- 语料、全库统计(N、avgdl、idf)与剪枝依据(块最大贡献)在一批中一起构建、原子生效。
- 更正后总篇数超过 5000 返回 `409`,旧状态完整保留;参数非法返回 `422`,旧状态不变。
- 写者串行:并发成功更正逐个生效、不丢失;读者不加锁:慢建索引期间查询仍用旧状态即时返回,
  构建成功后的新查询自动使用新状态(可用环境变量 `BM25_BUILD_DELAY=秒` 演示)。

### `POST /search` — 正式检索(块上界剪枝)

```json
{"terms": ["轴承过热", "更换轴承"], "k": 10}
```

响应:

```json
{
  "hits": [{"id": "R0001", "text": "...", "score": 3.42}],
  "scored": 88
}
```

- `hits`:命中记录的编号、正文、分数,按分数降序、同分编号升序,至多 k 篇。
- `scored`:完整计分篇数(通过枢轴检查、被完整累加分数的文档数)。
  块上界剪枝下 `scored` 通常远小于含词候选数与全库篇数——被跳过的候选未经完整计分。

### `POST /search/exhaustive` — 穷举对照(仅验证用)

全量计分后排序的对照接口,响应结构相同,`scored` 恒为全库篇数 N。
用于核对正式检索的前 k 一致性,不是正式检索路径。

## `POST /pairing/evaluate` — A/B 配对评估

比较同型号设备两种维修措施的历史恢复情况。提交 A、B 两组各 1~8 条**现存**维修记录编号,
以及逐单人工核实的恢复结果:

```json
{
  "a": [{"id": "R0005", "recovered": true}, {"id": "R0017", "recovered": false}],
  "b": [{"id": "R0028", "recovered": false}, {"id": "R0120", "recovered": true}]
}
```

规则:

- **相似度**:两组记录"去重后分词集合"的 Jaccard 系数。为空集合时相似度为 0。
- **可配对**:仅当相似度 ≥ 0.5 才允许 A、B 跨组配对;每条记录至多出现在一个配对中。
- **目标顺序**:先使配对数最多,再使相似度之和最大。
- **恢复率之差** = A 组恢复率 − B 组恢复率 = (A 恢复数 − B 恢复数) / 配对数。
  两组分母同为配对数。
- **多最优**:若达到上述目标的配对方案不止一种,响应同时给出差值的**最小值**与**最大值**
  及各自完整的配对明细,不会任选一种当作唯一结论;`unique_optimal_pairing` 标明是否唯一。
- **无配对**:相似度达到 0.5 的跨组配对一条也没有时,`insufficient_evidence = true`,
  报文说明"证据不足",`min_difference` / `max_difference` 为 `null`。
- **编号校验**:编号不存在、或在 A、B 内/跨组合并后重复,整次请求返回 `422`,
  响应体列出 `missing_ids` 与 `duplicate_ids`,不做任何评估。
- **同一版记录**:编号校验、配对与结论统计共用请求开始时取得的同一个不可变
  `State` 快照;响应中的 `version` 即该快照的版本号。成功更正后版本号递增,
  新请求按新版本重算;更正失败(如超上限 `409`)则版本号与记录都保持不变。

响应示例(多最优):

```json
{
  "version": 4,
  "paired": 1,
  "similarity_sum": "1/2",
  "optimal_pairing_count": 2,
  "insufficient_evidence": false,
  "unique_optimal_pairing": false,
  "min_difference": {
    "difference": "0", "difference_float": 0.0,
    "a_recovery_rate": "0", "b_recovery_rate": "0",
    "a_recovered": 0, "b_recovered": 0,
    "pairs": [{"a_id": "A2", "b_id": "B1", "similarity": "1/2", "similarity_float": 0.5}]
  },
  "max_difference": {
    "difference": "1", "difference_float": 1.0,
    "a_recovery_rate": "1", "b_recovery_rate": "0",
    "a_recovered": 1, "b_recovered": 0,
    "pairs": [{"a_id": "A1", "b_id": "B1", "similarity": "1/2", "similarity_float": 0.5}]
  },
  "message": "最优配对不唯一(共 2 种):恢复率之差最小 0、最大 1,须同时参考两者的配对明细"
}
```

分数以约分后的精确有理数(`"1/2"`、`"1"`、`"0"`)给出,同时提供浮点近似字段;
0.5 阈值与相似度之和均按精确值比较,不受浮点误差影响。
若多种最优方案的差值相同,`min_difference` / `max_difference` 各给出一份确定的代表明细。

### 其他

- `GET /stats`:篇数、平均长度、词项数、当前记录版本号、BM25 参数。
- `GET /healthz`:健康检查。
- `GET /docs`:Swagger UI(交互式 API 文档)。

## 剪枝正确性说明

每个词项的 posting 按 32 篇分块,预计算块内最大贡献;块最大贡献的后缀最大值即
"当前位置之后所有 posting 贡献"的上界。WAND 每轮按当前文档号排序游标,以前缀上界和
选择枢轴:上界和严格小于当前第 k 名分数的文档不可能进入前 k,按块整块跳过;
只有与枢轴文档对齐的候选才被完整计分。枢轴判断取 `>=` 阈值,同分文档仍可凭编号升序
进入前 k,保证与穷举逐位一致(同一 State、同一词项顺序累加,浮点结果完全相同)。

## 验证

```bash
python3 -m pytest tests/ -q          # 42 项单元/API/并发/配对测试
python3 scripts/verify.py            # 检索:穷举对照 + 随机回归 + 剪枝示例
python3 scripts/verify_pairing.py    # 配对:400 组随机用例与暴力枚举逐项比对
```

`scripts/verify.py` 用本地语料 `data/corpus.json` 与随机语料,将 BlockMax WAND 与
穷举实现、独立参照实现(直接按公式对原始语料计分)逐位比对,并打印完整计分篇数,
展示确有候选被跳过,例如:

```
查询 ['报警', '停机'], k=10:完整计分 88 篇;含查询词的文档 449 篇;全库 1500 篇
即 361 篇含词候选未经完整计分即被块上界跳过,前 10 名与穷举完全一致。
```

`scripts/verify_pairing.py` 用 400 组随机 A/B 组合与暴力枚举逐项比对最优配对数、
相似度之和、最优方案数、差值最小值/最大值及其明细,并验证 8×8 全同集合时
最优方案数为 8! = 40320。

## 目录

```
app/index.py      BM25 状态构建、BlockMax WAND、穷举对照、原子状态管理
app/pairing.py    配对相似度、最优配对、差值极值的精确计算
app/main.py       FastAPI 应用与端点
app/schemas.py    请求/响应模型
data/corpus.json  本地示例语料(scripts/gen_corpus.py 生成,确定性种子)
scripts/verify.py 检索的穷举对照与剪枝效果验证
scripts/verify_pairing.py 配对结果与暴力枚举的对照验证
scripts/load_corpus.py 向运行中的服务批量写入语料
tests/            单元、API、并发一致性与配对评估测试
```
