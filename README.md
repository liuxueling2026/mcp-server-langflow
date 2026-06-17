# Langflow MCP Server

通过 **MCP (Model Context Protocol)** 方式提供 Neo4j / PGVector 能力，替代 Langflow 中的自定义 Python 组件
（`components/neo4j_query.py`、`components/pgvector_combined.py`）。

这样在 **ICA 平台的 Langflow** 上就无需编写自定义 / Python 组件，只需用内置的「MCP Tools」节点连接本服务即可。

本服务暴露 **3 个工具**：

| Tool | 替代的自定义组件 | 说明 |
|------|------------------|------|
| `neo4j_query` | `neo4j_query.py` | 带参数的 Cypher 查询 |
| `pgvector_ingest` | `pgvector_combined.py`（写入） | 把字段 JSON 行做 embedding 并写入 PGVector，含去重 |
| `pgvector_search` | `pgvector_combined.py`（检索） | 带元数据过滤的向量检索 + 混合重排（向量 + trigram + 类型兼容） |

---

## 1. 架构

```
本地 MCP Server (HTTP :8000)  <-->  本地 Langflow (MCP Tools 节点)  <-->  Neo4j
            │                                                          PostgreSQL + pgvector
            └── Embedding Provider（Ollama 本地 / OpenAI 兼容，可切换）
```

- Transport：**Streamable HTTP**（FastMCP 默认，优于旧版 SSE）
- 连接密钥（Neo4j、PostgreSQL）与 Embedding 配置全部放在服务端 `.env`，**不**作为 tool 入参传递。
- **Embedding 可插拔**：换模型/换服务商只改 `.env`（`EMBEDDING_*`），业务代码零改动。


---

## 2. 安装

需要 Python ≥ 3.10。

```bash
cd mcp_server

# 方式 A: pip
pip install -e .

# 方式 B: uv（更快）
uv pip install -e .
```

---

## 3. 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
# ── Neo4j（neo4j_query 用）──
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io   # 或 bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的密码

# ── MCP Server ──
MCP_HOST=127.0.0.1
MCP_PORT=8000

# ── PostgreSQL / PGVector（pgvector_ingest / pgvector_search 用）──
# SQLAlchemy 风格连接串
PG_CONNECTION_STRING=postgresql+psycopg2://postgres:你的密码@localhost:5432/postgres

# ── Embedding Provider（可插拔，换模型只改这里）──
EMBEDDING_PROVIDER=ollama            # ollama | openai | watsonx
EMBEDDING_MODEL=nomic-embed-text     # 换 provider 时同步改：openai 用模型名 / watsonx 用 model_id
EMBEDDING_BASE_URL=http://localhost:11434

EMBEDDING_API_KEY=                   # openai / openai 兼容服务才需要
EMBEDDING_USE_TASK_PREFIX=false      # nomic 风格 search_document:/search_query: 前缀

# ── 仅当 EMBEDDING_PROVIDER=watsonx 时填写 ──
WATSONX_URL=https://jp-tok.ml.cloud.ibm.com
WATSONX_API_KEY=
WATSONX_PROJECT_ID=                  # project_id 与 space_id 二选一
# WATSONX_SPACE_ID=
```

> - 没填 Neo4j 也能启动服务（用于先验证 MCP 链路），但调用 `neo4j_query` 会返回连接错误。
> - **切换到公开模型**（如 OpenAI 或任意 OpenAI 兼容网关）只需改 `EMBEDDING_PROVIDER=openai` 并填 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL`，代码无需改动。
> - ⚠️ ingest 与 search **必须使用同一个 embedding 模型**，否则向量维度/语义不一致会导致检索失效。

#### 用 IBM watsonx.ai 做 embedding（provider=`watsonx`）

watsonx.ai **不兼容 OpenAI 协议**（用 IBM Cloud IAM 鉴权 + 必须的 `project_id`），因此单独走 `watsonx` provider（基于官方 `langchain-ibm` SDK，已在依赖中）。配置示例：

```dotenv
EMBEDDING_PROVIDER=watsonx
EMBEDDING_MODEL=ibm/granite-embedding-278m-multilingual   # 即 watsonx 的 model_id
WATSONX_URL=https://jp-tok.ml.cloud.ibm.com               # 你的 watsonx API Endpoint
WATSONX_API_KEY=你的-IBM-Cloud-API-Key
WATSONX_PROJECT_ID=d1117542-....                          # 你的 Project_ID
```

> - 对应 Langflow 里 watsonx Embedding 组件的字段：API Endpoint→`WATSONX_URL`、API Key→`WATSONX_API_KEY`、Project_ID→`WATSONX_PROJECT_ID`、Model Name→`EMBEDDING_MODEL`。
> - IAM 换 token、刷新等由 SDK 内部处理，无需手动管理。
> - ⚠️ **换 embedding 模型必须重新 ingest**：从 Ollama `nomic-embed-text` 换到 watsonx `granite-embedding-278m` 后，向量维度与语义空间均不同，旧 collection 不可复用——请用 watsonx 重新 `pgvector_ingest`（或写入新的 collection 名）。



---

## 4. 启动

```bash
python server.py
```

看到日志：
```
Starting MCP server on http://127.0.0.1:8000/mcp (Streamable HTTP transport)
```
即表示服务已就绪，MCP 端点为 **`http://127.0.0.1:8000/mcp`**。

---

## 5. 在本地 Langflow 中接入

1. 启动本地 Langflow，新建或打开一个 Flow。
2. 拖入 **「MCP Tools」** 组件（搜索 "MCP"）。
3. 选择连接模式 **SSE / HTTP**（不同 Langflow 版本名称略有差异），填入：
   - URL / SSE URL：`http://127.0.0.1:8000/mcp`
4. 连接成功后，组件会自动列出服务暴露的工具：**`neo4j_query`**、**`pgvector_ingest`**、**`pgvector_search`**。
5. 将该 MCP Tools 组件接入 Agent，或直接由下游节点调用对应工具。


---

## 6. `neo4j_query` 工具说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `cypher` | string | ✅ | Cypher 语句，支持 `$param` 占位符；多语句用分号分隔 |
| `params` | object | ❌ | `$param` 对应的参数字典 |

返回：
```json
{
  "records": [ { "...": "..." } ],
  "count": 3
}
```
- Neo4j **Node** 会带 `_labels` 字段；**Relationship** 会带 `_type` 字段；
- 时间类型自动转 ISO 字符串。
- 出错时返回 `{"error": "...", "records": [], "count": 0}`。

### 参数化查询示例

```jsonc
// cypher
"MATCH (s:System {name: $system})-[:HAS_OBJECT]->(o:Object {name: $name}) RETURN o"

// params
{ "system": "Veeva", "name": "Account" }
```

### 快速验证（无需 Langflow）

```bash
python selftest.py
```
该脚本会校验配置、测试 Record→JSON 序列化逻辑；若 `.env` 已配置可连的 Neo4j，还会跑一条真实查询。

---

## 7. `pgvector_ingest` 工具说明

把字段 JSON 行做 embedding 写入指定 collection。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `rows` | string / array / object | ✅ | — | 字段行。支持 **JSON 数组字符串**、单对象字符串、list/dict |
| `collection` | string | ✅ | — | 目标 collection（表）名 |
| `dedup_mode` | string | ❌ | `by_object_field` | `none` / `pre_delete_all` / `by_object_field` / `by_content_hash`，删除均限定在该 collection |
| `use_task_prefix` | bool | ❌ | 跟随 `.env` | 覆盖 nomic 前缀开关 |

**每行 JSON 的字段** → 写入的 metadata 映射：

| 行内 key | metadata | 说明 |
|----------|----------|------|
| `system` / `object` / `objectdescription` / `company` | `system` / `object` / `object_description` / `company` | |
| `fieldname` | `field` | |
| `fieldlabel` | `field_label` | 参与 embedding 文本与 trigram |
| `datatype` | `field_type` | 参与类型兼容重排 |
| `fielddescription` | `description` | 参与 embedding 文本 |

> embedding 文本 = `"{field_label}. {description}"`；缺 `object` 或 `fieldname` 的行会被跳过。

**Langflow 连法**（已验证）：用 Text Operation / Prompt Template 把上游多行拼成**标准 JSON 数组字符串** `[{...},{...}]`，连到工具的 `rows` 端口即可。ingest 这条路径**无需** Split Text（每行本身就是一条完整字段记录）。

`rows` 输入示例：
```json
[{"system":"LSC","object":"ProviderAffiliation","company":"abbott",
  "fieldname":"LastViewedDate","fieldlabel":"Last Viewed Date",
  "fielddescription":"The timestamp ...","datatype":"datetime"},
 {"system":"LSC","object":"HealthcareProviderSpecialty", "...": "..."}]
```
返回：`{"ingested": 2, "dedup_mode": "by_object_field"}`

---

## 8. `pgvector_search` 工具说明

带元数据过滤的向量检索，再做**混合重排**：向量相似度 + trigram 字符串相似度（对 `field_label`/`field`）+ Veeva→LSC 数据类型兼容度。

支持两种入参风格，任选其一：

**① 统一 `search_data`（推荐，1:1 对齐原组件「Search Data 端口」）**

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `collection` | string | ✅ | — | 要检索的 collection（须与 ingest 一致） |
| `search_data` | object / string | ✅* | — | 统一 JSON，见下例。`candidates` 可用单引号 |
| `search_query_key` | string | ❌ | `searchQuery` | 从 search_data 取 query 的 key |
| `filter_key` | string | ❌ | `candidates` | 取过滤 candidates 的 key |
| `field_label_key` | string | ❌ | `sourceFieldLabel` | 取 field_label 的 key |
| `field_type_key` | string | ❌ | `sourceFieldType` | 取 field_type 的 key |

`search_data` 示例（直接连原 workflow 的 Parser 输出）：
```json
{"searchQuery": "Best Phone Number. n/a",
 "candidates": [{"system":"LSC","object":"Account"},
                {"system":"LSC","object":"ContactPointPhone"}],
 "sourceFieldLabel": "Best Phone Number",
 "sourceFieldType": "Phone"}
```
内部会把 candidates 归一化为 `{"system":"LSC","object":["Account","ContactPointPhone"]}` 做多值过滤。

**② 离散参数**（程序化 / LLM 直接调用；设置后优先于 search_data）

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | string | 检索文本 |
| `filter` | object / string | 支持 `{"object":"A"}`、多值 `{"object":["A","B"]}`、Neo4j `{"result":[{"candidates":[...]}]}`、裸 `[{system,object},...]` |
| `field_label` | string | 用于 trigram 重排 |
| `field_type` | string | 用于类型兼容重排 |

**通用调参**

| 参数 | 默认 | 说明 |
|------|------|------|
| `number_of_results` | 5 | 重排后返回条数 |
| `recall_top_k` | 50 | 重排前向量召回条数 |
| `vector_weight` / `trigram_weight` / `type_weight` | 0.5 / 0.35 / 0.15 | 重排权重 |
| `type_mapping` | 内置 Veeva→LSC | 可传 dict/JSON 覆盖 |

返回：
```json
{"results": [{"text": "...", "metadata": {"object": "...", "field": "...", "final_score": 0.83}}],
 "count": 1}
```
> 若提供了 filter 但解析失败，返回 `count:0` 并带 `error`，避免无过滤的全表泄漏。

---

## 9. 备注

- ingest 与 search 共用同一进程、同一 embedding provider；端口被占用（`Errno 10048`）时先停掉旧 `server.py` 进程再启动。
- 连接密钥不经 tool 入参，符合 ICA 的密钥管理模型。

```
