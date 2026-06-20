# MVW 变更记录

这份记录只描述**当前工作树中已经实现并已验证**的内容，不把推测、设计意图或未覆盖的结论写成既成事实。

当前验证命令：

```bash
./.venv/bin/pytest tests/test_mvw.py tests/test_mvw_eval.py tests/test_mvw_learning.py -q
```

当前结果：**18 passed in 157.77s**

---

## 1. magic_box 从 Python 注入改为 KB 原生规则

### 当前状态

`magic_box` 的 `golden(...)` / `transformed(...)` facts 已经不再由 `apply_novelty_runtime` 在 `env.step()` 后手动注入，而是由 TextWorld KB 中的规则直接产出。

### 相关文件

- `textworld/mvw/data/logic/magic_box.twl`
- `textworld/mvw/kb.py`
- `textworld/mvw/curriculum.py`
- `textworld/mvw/scenarios.py`
- `textworld/envs/tw.py`

### 实际实现

- 新增 `load_magic_box_kb()`，加载内置逻辑加 `magic_box.twl`。
- `stage_5` 的 magic-box 分支现在使用 `type="magic_box"`，不再退回普通 `c` 容器。
- `apply_novelty_runtime(...)` 仍然保留为场景钩子，但对 `magic_box` 已经是 no-op。
- `textworld/envs/tw.py` 对 `.json` 环境的 `_valid_actions` 按 precondition 数量降序排序，并在 `step()` 中按这个排序后的 action 列表回取动作。这个改动是为了在命令字符串相同的情况下优先命中更具体的 `open/magic_box` 规则。

### 为什么需要这个改动

原先 `portal` 有 `portal.twl`，但 `magic_box` 只有 Python 层后处理，世界并不闭合。现在 `magic_box` 至少和 `portal` 一样，进入了 KB / transition rule 层，而不是只在评测脚本里看起来成立。

---

## 2. planner 现在真正可插拔，并贯穿 benchmark

### 当前状态

`evaluate_game(...)`、`evaluate_counterfactuals(...)`、`evaluate_rule_minimality(...)`、`discover_rule_patches(...)`、`evaluate_patch_transfer(...)`、`plan_with_model(...)`、`evaluate_planning_improvement(...)`、`evaluate_benchmark(...)` 都支持传入 `planner`。

此外，`evaluate_novelty_accommodation(...)` 现在也会把 `planner` 传到底层 `evaluate_game(...)`，所以 `evaluate_benchmark(planner=...)` 不再是“部分 data-driven、部分 rule-based”的混合执行。

### 相关文件

- `textworld/mvw/runner.py`
- `textworld/mvw/eval.py`
- `tests/test_mvw_eval.py`

### 额外修复

`textworld/mvw/runner.py` 里曾经存在一整段重复定义，后半段旧实现会覆盖前半段带 `planner=` 的版本。现在已经收束成单一实现，避免“文档说支持 planner，但运行时实际没走到”的情况。

### 验证

现有测试不仅检查 `DataDrivenExpansionPlanner()` 的 benchmark 指标，还显式检查 accommodation trace 中实际使用的 patch：

- `portal` 路径应看到 `portal_transition`
- `magic_box` 路径应看到 `transform_on_open`

这能防止 benchmark 在某一环悄悄掉回默认 `RuleBasedExpansionPlanner`。

---

## 3. DataDrivenExpansionPlanner 是“信号驱动模板归纳”，不是完全无先验

### 当前状态

仓库里确实新增了 `DataDrivenExpansionPlanner`，但它不是“完全从数据中自由归纳”的 planner，也不应该被描述成“无 hardcoded knowledge”。

### 实际行为

它仍然依赖一些轻量结构假设：

- `unsupported_action + verb == use` 时归为 `portal_transition`
- `verb == open` 且出现一元 property missing facts 时归为 `transform_on_open`
- 其余情况退化为根据 missing / unexpected facts 生成 generic patch

### 相关文件

- `textworld/mvw/models.py`

### 更准确的表述

更准确的说法是：

> `DataDrivenExpansionPlanner` 比 `RuleBasedExpansionPlanner` 少依赖具体实体名，但仍然是模板化、带结构先验的 signal-driven baseline，而不是通用规则归纳器。

这个版本仍然有价值，因为它把 novelty 归纳从“认 portal / 认 gold 关键词”推进到了“看命令结构 + 事实差分”，但还没有到 paper 级的开放式 rule induction。

---

## 4. 增量学习补上了 update 路径，但 BeliefTracker 和 TransitionModel 的语义不同

### 当前状态

学习模块现在支持：

- `_state_signal(record)`
- `LinearMultiLabelModel.partial_fit(...)`
- `BeliefTrackerModel.update(...)`
- `TransitionModel.update(...)`
- `summarize_incremental_update(...)`

### 相关文件

- `textworld/mvw/learning.py`
- `tests/test_mvw_learning.py`

### 需要明确的语义

- `TransitionModel.update(...)` 是真正的增量更新路径。它扩展 action / fact 词表后，对新数据做 `partial_fit`，并冻结旧输出列来减少遗忘。
- `BeliefTrackerModel.update(...)` 存在，但跨 stage 语义并不稳妥。`summarize_incremental_update(...)` 里对 BeliefTracker 的处理仍然是**在 base + new 上重新 fit**，这是当前代码中更可信的用法。

也就是说，当前实现支持“增量更新实验”，但不能把它描述成“两个模型都完成了完全对称、同等可靠的 continual-learning 方案”。

---

## 5. magic_box 不再固定是 apple，补丁转移现在是跨 seed、跨目标物体验证

### 当前状态

`magic_box` 现在会根据 seed 选择不同目标物体，当前集合是：

- `apple`
- `carrot`
- `potato`

`evaluate_patch_transfer(...)` 会先在一个 seed 上发现 patch，再把 patch 直接拿到另一个 seed 上复用。

### 相关文件

- `textworld/mvw/scenarios.py`
- `textworld/mvw/curriculum.py`
- `textworld/mvw/eval.py`
- `tests/test_mvw_eval.py`

### 为什么这比之前更好

这样至少能区分：

- “只记住 `open magic box -> golden(apple)` 这个 episode”
- 和
- “学到 `open container -> transform contained object` 这种可迁移规则”

当前测试已经覆盖：

- `seed=2026` 和 `seed=2027` 的目标物体不同
- 迁移前失败、迁移后成功

---

## 6. novelty benchmark 现在不止两类场景

### 当前状态

`stage_5` 现在包含三类 novelty family：

- `portal`
- `magic_box`
- `bridge_button`

### 相关文件

- `textworld/mvw/curriculum.py`
- `textworld/mvw/scenarios.py`
- `textworld/mvw/data/logic/bridge_button.twl`
- `tests/test_mvw_eval.py`

### 为什么这一步重要

`bridge_button` 引入的是第三种机制：不是 teleport，也不是 property transform，而是**新 action 释放一条原本被阻塞的 map edge**。这让 benchmark 至少覆盖了：

- 新 transition edge (`portal`)
- 新 object-property transform (`magic_box`)
- 新 control action / path activation (`bridge_button`)

这仍然远远不够顶会规模，但比“两种手工 novelty demo”更接近一个系统化 family benchmark。

---

## 7. SearchExpansionPlanner 和 ablation report 已接入主路径

### 当前状态

仓库里现在有第三个 planner：

- `SearchExpansionPlanner`

它不是直接返回模板，而是对候选 patch 做打分：

- 当前 novelty 解释误差
- 已见 replay transition 保真
- patch complexity

同时新增了：

- `evaluate_novelty_suite(...)`
- `generate_ablation_report(...)`
- `format_ablation_markdown(...)`
- CLI: `scripts/tw-mvw report`

### 相关文件

- `textworld/mvw/models.py`
- `textworld/mvw/eval.py`
- `textworld/mvw/report.py`
- `scripts/tw-mvw`
- `tests/test_mvw_eval.py`

### 当前能确认的行为

直接运行：

```bash
./.venv/bin/python scripts/tw-mvw report --known-stage stage_4 --seed 2026 --format markdown
```

会输出 scenario × planner 的指标矩阵，当前覆盖：

- scenarios: `portal`, `magic_box`, `bridge_button`
- planners: `rule_based`, `data_driven`, `search`

在当前实现上，`search` 在 `magic_box` 上会选回 complexity 更低的 patch，因此 `rule_minimality` 从 data-driven 的 `4` 回到 `2`。

---

## 8. 当前能确定的 benchmark 结论

基于现有测试和直接 runtime 检查，可以确定：

- `portal`、`magic_box`、`bridge_button` 三条 `stage_5` novelty 路径都能通过 `idea.md` 的 6 项主指标评测。
- `magic_box` 的 patch transfer 已被纳入 benchmark details。
- `bridge_button` 的 patch transfer 和 planning improvement 也已纳入 benchmark details。
- `DataDrivenExpansionPlanner` 在 `magic_box` 上的 `rule_minimality` 高于 `RuleBasedExpansionPlanner`。
- `SearchExpansionPlanner` 已经接入 benchmark 主路径，而不是只存在于独立 helper 中。

对当前工作树，直接检查 `DataDrivenExpansionPlanner + magic_box` 可见：

- accommodation trace 的 patch 是 `transform_on_open`
- `rule_minimality == 4`

这比 rule-based 的 `2` 更高，是因为 data-driven 版本会显式记录两个新增 property：

- `golden`
- `transformed`

---

## 9. 仍然存在的限制

这些限制是当前实现里真实存在的，不应该在变更记录里被省掉：

- `DataDrivenExpansionPlanner` 仍然是模板化 baseline，不是开放式规则发现器。
- `magic_box` 的原生化依赖 `.json` 环境里对 `_valid_actions` 的排序修复；这是一个工程性 workaround，不是底层 TextWorld rule-selection 的根本重构。
- `BeliefTrackerModel.update(...)` 虽然存在，但跨 stage 的最稳妥路径仍然是 refit on merged data。
- benchmark 现在更接近 `idea.md`，但还没有把 novelty 完全下沉成统一的世界本体扩展机制；当前仍是三类显式场景。
- `SearchExpansionPlanner` 虽然比模板 planner 更接近“搜索式扩展”，但候选集仍然来源于人工设计的 patch family，不是开放式 hypothesis space。

---

## 10. 这份记录刻意没有写的内容

以下内容如果没有再次实测，我不把它写进这里：

- 具体 benchmark 分数表
- 具体学习曲线数字
- “纯数据驱动”“完全无 hardcode”“端到端规则归纳”这类过强表述

如果后续要把这些数字放回 `CHANGES.md`，应该先附上对应命令和当次输出，而不是凭印象补表。
