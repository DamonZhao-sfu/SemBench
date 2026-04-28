# BARGAIN-R 实现计划（先写计划，不改代码）

## 0) 目标与约束
- 仅实现 **BARGAIN-R**（Recall 约束版本），不混入 BARGAIN-A / BARGAIN-P 的功能。
- Proxy 固定为 **Logistic Regression**（LR）。
- Oracle 从“高成本 LLM”替换为 **correct+smooth** 判定器（即对候选标签执行 correct，再做 smooth 后得到最终 oracle 标签/分数）。
- 置信度计算与阈值选择遵循 BARGAIN 的核心思想：
  1) 用 proxy 先给每条记录打分；
  2) 通过抽样+统计下界，选出可保证 recall 目标的阈值；
  3) 在线阶段：高置信走 proxy，低置信升级到 oracle。
- 明确规避 **self-evaluation bias**：不能用同一批样本既训练 LR、又评估其可靠性并用于阈值担保。

---

## 1) 先对齐 BARGAIN-R 的对象定义（实现前统一符号）
对每条样本 \(x_i\)：
- \(\hat y_i^{proxy}\)：LR 的预测标签。
- \(s_i\)：LR 的置信度分数（下文定义）。
- \(\hat y_i^{oracle}\)：correct+smooth 的输出（视为高质量参考）。

针对 recall 任务（以正类为关心对象）：
- “proxy 接管”条件：\(s_i \ge t\)（阈值 \(t\)）。
- 否则回退 oracle。
- 整体系统输出与 oracle 输出比较，需满足目标 recall \(\tau\)（如 0.9）并带失败概率 \(\delta\)。

---

## 2) LR confidence 怎么算（核心回答）
对于二分类 LR，模型给出：
\[
p_i = P(Y=1\mid x_i)
\]

建议使用 **margin-based confidence**（最贴合 BARGAIN 里“越大越可信”的单调分数要求）：
\[
conf_i = \max(p_i, 1-p_i) = \tfrac{1}{2}+\left|p_i-\tfrac{1}{2}\right|
\]
等价 margin 形式：
\[
m_i = |2p_i-1|,\quad conf_i = \tfrac{1+m_i}{2}
\]

### 为什么用这个
- 与预测标签一致：\(\hat y_i^{proxy}=\mathbb{1}[p_i\ge 0.5]\) 时，\(conf_i\) 正好表示“离决策边界有多远”。
- 单调可排序：适合阈值扫描（BARGAIN 的阈值选择依赖排序/分桶/前缀集合）。
- 比直接用 logit 更直观，且可选后续做校准（Platt/Isotonic）。

### 若是多分类（预留）
\[
conf_i = \max_k P(Y=k\mid x_i)
\]
并可配合 top1-top2 gap 作为替代分数；本次先按二分类实现。

---

## 3) 避免 self-evaluation bias 的数据协议（必须做）
采用“训练/校准/担保”三段式隔离：

1. **Train split**：只用于训练 LR 参数。
2. **Calibration split**：只用于分数校准（可选）+ 早期阈值候选筛选，不做最终担保结论。
3. **Guarantee split**：只用于最终统计保证（估计 recall 下界并选最终阈值）。

关键原则：
- 任何用于“理论保证”的统计量，必须来自 **未参与 LR 拟合** 的样本。
- 若样本很少，使用 **K-fold cross-fitting**：
  - 每折在其余折训练 LR；
  - 对留出折打 out-of-fold 预测分数；
  - 聚合全部 OOF 分数后再做阈值担保。
这样可显著降低乐观偏差。

---

## 4) BARGAIN-R 阈值选择（按 paper 思路落地）
> 不写代码版本，先定流程。

1. 在 guarantee split 上，收集每条样本的：
   - proxy 预测（由 LR 产生）
   - proxy confidence \(conf_i\)
   - oracle 标签（correct+smooth）

2. 以 \(conf_i\) 从高到低排序，构造阈值候选集 \(\{t_j\}\)。

3. 对每个阈值 \(t_j\)，计算“若 \(conf_i\ge t_j\) 则交给 proxy，否则交给 oracle”的**经验 recall**（相对 oracle 正类）。

4. 对每个阈值的 recall 用 **一侧置信下界**（Clopper-Pearson / Wilson / paper中同款界）做保守修正，得到 \(LB_j\)。

5. 选择最大化 proxy 覆盖率且满足
\[
LB_j \ge \tau
\]
的阈值 \(t^*\)。若都不满足，退化为全 oracle。

6. 在线推理时固定 \(t^*\)，不可再用在线样本回头调阈值（避免选择偏差）。

---

## 5) correct+smooth 作为 oracle 的接口设计
- 统一 oracle API：`oracle.predict(x)` 返回最终标签，`oracle.score(x)`（可选）返回平滑后的置信。
- 为 BARGAIN-R 的担保，最低需求是稳定标签输出；若 recall 定义依赖概率阈值，则固定该阈值并在配置中显式记录。
- correct+smooth 的内部随机性需可控（固定 seed 或 deterministic 模式），避免担保集评估噪声。

---

## 6) 实施步骤（真正写代码时按此顺序）
1. 梳理现有 pipeline 中 proxy/oracle 抽象，确认 BARGAIN-R 插入点。
2. 增加 LR proxy 适配层：训练、`predict_proba`、`confidence`。
3. 增加 correct+smooth oracle 适配层。
4. 增加 split/cross-fitting 机制，强制训练与担保解耦。
5. 实现阈值扫描 + recall 一侧下界计算 + 阈值选择器。
6. 加入失败回退逻辑（全 oracle）。
7. 增加实验日志：阈值、覆盖率、经验 recall、下界 recall、delta、seed。
8. 单测与小规模端到端验证。

---

## 7) 验收标准（DoD）
- 在固定数据与 seed 下可复现同一阈值与结果。
- 在 guarantee split 上：报告的 recall 下界满足目标 \(\tau\)（置信度 \(1-\delta\)）。
- 与“全 oracle”对比，proxy 调用比例提升且不破坏 recall 约束。
- 明确证明没有 self-evaluation bias（通过 split 或 cross-fitting 日志可审计）。

---

## 8) 需要你确认的实现细节（编码前）
1. 正类定义（recall 针对哪一类）。
2. correct+smooth 的具体输入输出格式与阈值。
3. 统计下界你更偏好哪一种（paper 同款优先；否则 Wilson 一侧下界）。
4. 数据量是否足够三段切分；不足则默认 cross-fitting。

---

## 9) Detailed TODO Checklist（执行状态）

### Phase 1: 规格冻结
- [x] 明确仅实现 BARGAIN-R。
- [x] 明确 Proxy=Logistic Regression。
- [x] 明确 Oracle=correct+smooth。
- [x] 明确 recall 目标 \(\tau\) 与失败概率 \(\delta\) 作为运行时参数。

### Phase 2: 核心组件实现
- [x] 实现 LR 训练与 `predict_proba`。
- [x] 实现 LR `confidence = max(p, 1-p)`。
- [x] 实现 correct+smooth oracle 的 `score` / `predict`。
- [x] 实现 BARGAIN-R 阈值候选扫描。
- [x] 实现 Wilson 一侧下界计算。
- [x] 实现满足 \(LB\ge\tau\) 的最大覆盖率阈值选择。

### Phase 3: 无偏评估协议落地
- [x] 切分训练集与担保集（训练/担保隔离）。
- [x] 在担保集计算阈值担保统计量。
- [x] 禁止在线阶段回调阈值。

### Phase 4: 可运行脚本与输入输出
- [x] 提供 CLI（输入文件、特征列、标签列、oracle 分数字段、\(\tau\)、\(\delta\)）。
- [x] 输出阈值、覆盖率、经验 recall、recall 下界。
- [x] 保持类型注解完整，不使用 `Any` / `unknown` 风格占位类型。

### Phase 5: Supporting Queries 可视化
- [x] 提供图1：Recall Target vs Precision。
- [x] 提供图2：Recall Target vs Calls Saved。
- [x] 输出按 recall target 聚合表。

### Phase 6: 自检与收尾
- [x] 运行语法/类型检查（受环境依赖约束时记录）。
- [x] 更新计划文档任务状态为完成。
- [x] 提交代码与变更说明。
