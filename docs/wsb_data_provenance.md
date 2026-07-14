# Workspace-Bench 数据真实性核查（2026-07-09）

针对"WSB 是真实数据还是也存在造的数据"的验证。**结论：混合来源，
既非纯真实企业数据，也非纯合成——他们自己在论文里就说清楚了。**
以下是证据。

## 一、WSB 论文原文的自述（arXiv 2605.03596v4）

论文 §2 "Workspace Construction" 明确写了 pipeline 是
**"hybrid file collection and generation"**（混合式收集与生成）。
关键句摘录：

> "To ensure both realism and reproducibility, we construct
> Workspace-Bench through a controlled pipeline that combines
> persona-driven workspace simulation, hybrid file collection and
> generation, task curation, dependency annotation, and expert
> validation."

> "After constructing the directory, we populate each workspace using
> a **hybrid strategy that combines real-world data retrieval and
> grounded generation**. We first deploy a semantic-driven agentic
> crawler that traverses the generated directory tree and retrieves
> public resources relevant to the semantic workspace directory,
> such as arXiv papers, GitHub repositories, technical documents,
> reports, spreadsheets, and presentation materials. **We then use
> LLMs to synthesize related artifacts grounded in the collected
> files**, such as emails discussing a paper, meeting notes referring
> to a design document, or reports derived from spreadsheets."

**关于任务来源**（这个是真的）：

> "We conducted an in-depth analysis of **154 authentic task scenarios
> sourced from the Lark platform in ByteDance**."

> "25 human annotators aligned with the five workspace roles create
> concrete tasks within the simulated workspaces. For each task,
> annotators write the natural language instruction, identify the
> required inputs, produce a reference output, and design evaluation
> rubrics."

## 二、Lite-en 文件抽检证据（直接看下载下来的数据）

### 证据 1：Task 3 (Backend Developer) 的 35 个 dependency_item_*.md
是模板化生成的

对同一 task 内两个不同文件做 diff：

```
$ diff dependency_item_12.md dependency_item_33.md
1c1
< # Dependency Management Report 12
---
> # Dependency Management Report 33
4c4
< This document tracks all dependencies for module 12.
---
> This document tracks all dependencies for module 33.
```

**除了模块编号，逐字相同**。35 个"依赖清单文件"内容全部一样，只是
编号 1-35。这是典型的 LLM 模板批量生成，不是真实项目文档。

35 个文件的确都有不同的 md5 hash（因为编号不同），但内容 99% 重复。

### 证据 2：Task 143 (Operations Manager) 的社交媒体 post JSON 是
合成的

抽两个 post：

```json
POST_91: caption="Check out our latest update and exclusive offer! 🎉
                  #product_91 #marketing #technology"
POST_53: caption="Check out our latest update and exclusive offer! 🎉
                  #product_53 #marketing #technology"
```

caption 字段除数字外一模一样。数据字段（likes、reach）看起来是随机
生成的，还有明显的合成错误：`post_date: "2024-02-08T023:18:00Z"`（
时刻字段的小时位是 023，非法 ISO 8601）。

### 证据 3：Task 7 的会议纪要看起来像真实模板但内容合成

`2024-12-project-kickoff-meeting-minutes.md` 内容格式规整，包含
"Project Manager Wang Ming"、"Backend Developer Zhang San" 等角色名
（"Wang Ming/Zhang San/Li Si/Wang Wu" 是典型的合成占位人名，中文语
境里等价于"张三李四"）。内容本身看起来是真实企业会议纪要的模板+LLM
填充。

### 证据 4：Task 15 (Logistics/Operations) 的财务表格 xlsx 结构真实
但数据造的

```
Sheet: 'Sheet1'
row 2: ('Month', 'Income Amount', 'Cost Amount', ...)
row 3: ('Jan', 3200, 1200, 200, 'Q1')
row 4: ('Feb', 2800, 900, 300, 'Q2')
row 5: ('Mar', 3500, 1100, 400, 'Q3')
```

数字都是整百整千的整数，货币单位不明。真实企业 P&L 报表不会这样出。
这类小规模数字整齐的表格典型是 LLM prompt 里让它 "make up income and
expenses for 12 months" 生成的。

### 证据 5：Task 128 的 Python 代码 `.py` 文件

Task 128 的输入是 5 个真实 Python 文件（`table_preprocess.py`、
`gradio_app.py` 等），来自开源项目 "ST-Raptor"。这符合论文里说的
"agentic crawler retrieves public resources"——**代码文件是真爬来的**。

## 三、结论：三层混合来源

| 数据层 | 来源 | 真实度 |
| --- | --- | --- |
| **任务描述 (task instruction)** | 154 个真实 Lark/ByteDance 内部场景抽象后由 25 名标注员改写 | **真实场景**（不是真实数据） |
| **File dependency graph** | 标注员手工标注 + expert 校验 | **人标注真实** |
| **Rubrics** | 标注员写、专家校验 | **人写真实** |
| **代码文件、arXiv 论文、GitHub 资源** | 语义 crawler 从公开互联网爬取 | **真实文件** |
| **文档 (.md, .docx)、报告、会议纪要、社交 post、假名单** | LLM 生成，grounded in 上一层文件 | **LLM 生成** |
| **表格 (.xlsx, .csv)** | LLM 生成 grounded in 场景，格式模仿真实业务表 | **LLM 生成** |

## 四、这对 WorkSurface-Bench 论文的影响

### 4.1 不能宣称"real enterprise data"，但能宣称什么

**不能说**：
- "real enterprise data"
- "authentic company documents"
- "human-authored files"

**可以说**：
- "**enterprise-grounded**"（场景真实，数据 LLM-synthesized-from-real-scenarios）
- "**real task scenarios from Lark/ByteDance**, with hybrid retrieved+generated file artifacts"
- "**expert-validated** task instructions, rubrics, and dependency graphs"

WSB 团队自己就是这么措辞的，我们照抄不错。这是学术界公认的合规做
法（AgentBench、TheAgentCompany 也是合成 workspace + 真实场景）。

### 4.2 我们的差异化叙事**几乎不受影响**

回顾 story：
- 主张 1「多面路由未被评过」→ **成立**，跟数据合成度无关
- 主张 2「workspace-derived 源数据」→ **仍成立**，把 "workspace-
  derived" 明确为 "enterprise-scenario-grounded, hybrid-generated
  files" 即可；反正 HybridQA/OTT-QA 也是 Wikipedia 拼的，比合成还
  合成
- 主张 3「Route ≠ Answer」→ 完全不受影响

### 4.3 需要在论文里正面回应的地方

在 Related Work / Method 章节加一段 provenance disclaimer：

> "WorkSurface-Bench inherits Workspace-Bench's hybrid data-construction
> pipeline: task scenarios and dependency graphs are human-authored and
> expert-validated from real Lark/ByteDance workflows, while file
> contents combine public web resources (papers, repositories, official
> reports) with LLM-generated artifacts grounded in the collected files.
> The Route / Evidence / Answer decomposition we introduce operates on
> the same artifacts regardless of provenance, and our contamination
> hygiene protocol (§X) treats LLM-generated content as potentially
> memorized by evaluated models."

**这段话的作用**：
1. 诚实披露，reviewer 不能追着问
2. 把污染这件事和 hybrid 数据源自然衔接起来（我们本来就有闭卷 probe）
3. 强调"Route/Evidence/Answer 分解是数据-provenance-agnostic 的方法
   贡献"，把审稿人注意力从"数据造得如何"引向"评测协议如何"

## 五、审稿人可能追问的问题 & 应对

Q1: **"合成数据的 rubric 靠不靠得住？"**  
A: 25 名标注员写 + 20 名 domain expert 交叉验证，WSB 论文 §3 报告
Cohen κ = 0.72（尚可）。我们的 atomic tasks 派生保留 wsb_commit +
rubric_refs 溯源，评测代码在 issue 时可追责。

Q2: **"LLM 生成的文档会不会被评测的 LLM 训练时见过？"**  
A: 这是合法的 concern。我们的 closed-book contamination probe 就是
干这个的（§C5）。probe 报告 > 20% 命中的 model 会打旗，读者自己判
断。

Q3: **"你们比 Workspace-Bench 提出的新东西是什么？既然数据是从他
们那儿来的？"**  
A: 我们不重造数据；我们提出一种**评测协议**：把同一 workspace 投
影成三个可路由 surface，独立打 Route 分数。这是 evaluation
methodology 贡献，不是 dataset scale 贡献。

## 六、行动项

- [x] 弄清 WSB 数据真实性并存档（本文件）
- [ ] 在 `paper_spec_zh.md` C2 加"provenance disclaimer" 段落  
- [ ] 在 `related_work_zh.md` WSB 对比条目里改措辞（不用 "real
       enterprise data"，改 "enterprise-scenario-grounded hybrid
       data"）
- [ ] Pilot 时对每 task 派生做一个字段：`data_provenance:
       {"scenario": "human-authored", "files": "hybrid (crawled +
       LLM-generated)"}`，写进 schema 也是诚实的做法
