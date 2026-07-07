# 面向多镜头生成视频的场景连续性诊断 Benchmark Proposal

## 1. 背景与动机

真实影视创作并不是由彼此独立的单镜头组成。一个场景通常通过多个镜头完成一段连续叙事：远景建立空间，近景呈现人物反应，特写突出关键道具，正反打推进对话，插入镜头补充动作细节，再回到远景重新确认空间关系。在这个过程中，镜头的景别、机位和构图不断变化，但观众默认这些镜头发生在同一个连续的时空中。

因此，影视创作对"连续性"有天然要求。同一场景中的人物外观、服装、道具状态、背景陈设、空间布局和光影条件，都需要在剪辑之间保持自洽。一个杯子被放在桌上后，后续镜头中不应无故变色或消失；墙上的画、桌边的灯、窗户和门的位置关系不应在正反打之间随意变化。这些问题在传统影视制作中被称为 continuity errors，也就是穿帮或连续性错误。

随着视频生成模型从单镜头短片段走向多镜头叙事生成，它们也开始面对类似要求：模型不仅要让每个镜头都合理、清晰、好看，还要让多个镜头组合起来像发生在同一个连续场景中。仅评估单镜头质量或主体一致性已不足以衡量这种能力——一个视频即使主角始终一致，也可能由于道具、背景、空间或光影的不连续而破坏叙事可信度。

因此本 benchmark 的核心出发点是：

**主角一致不等于场景连续。**

我们要评估的是：生成模型能否在一个较长的多镜头场景中，同时保持创作者明确要求的内容连续，以及模型自己建立的场景世界连续。

## 2. 核心问题定义

### 2.1 两条互补的连续性

我们将多镜头影视连续性定义为两种互补能力：

1. **Prompt-grounded Continuity**：caption / shot prompt 中明确提到的人物、物体、状态和关系，应在相关镜头中被正确生成并跨镜保持稳定。它衡量模型能否遵守创作者明确给出的连续性约束。
2. **Intrinsic Self-Consistency**：prompt 未明确规定、但模型在画面中自己建立的场景细节，一旦成为同一场景的一部分，也应在后续可比镜头中与自身保持自洽。它衡量模型能否维护自己生成出的场景世界。

这两者共同构成模型在影视创作中的连续性能力。前者对应剧本和分镜中明确要求的主体、关键道具与动作；后者对应观众在观看多镜头场景时自然建立的空间和环境记忆，例如灯、窗、墙画、沙发、整体光影和场景布局。

### 2.2 输入设定：video + caption，全自动

Benchmark 的输入是：

- 生成视频；
- episode / shot-level caption；
- 从 caption 中自动提取的主动提及主体列表（人物、物体、地点、动作状态和显式关系），作为 prompt-grounded track 的候选检查对象。

我们不假设人工标注的 continuity contract，也不使用参考视频。评测过程完全自动。caption 只负责提供外部要求和镜头语义，模型自由生成出的额外场景信息则由评测器从视频中在线分析。

### 2.3 与普通 prompt-following 的区别

Prompt-grounded continuity 不是简单的单镜头 prompt-following。它包含两个层次：

- **Shot realization / coverage**：caption 明确要求的主体或物体是否在应出现的镜头中被生成并可定位；
- **Cross-shot stability**：这些已生成的 prompt-grounded elements 是否在多个镜头中保持外观、身份、状态和关系稳定。

因此，"prompt 中要求蓝杯子但第一镜没有杯子"是 coverage / realization failure；"第一镜生成了蓝杯子，后续可比镜头变成白杯子"是 continuity failure。两者都重要，但需要分开报告，避免把 prompt-following 和跨镜连续性混成一个黑箱分数。

## 3. Benchmark 定位与贡献

本 benchmark 定位为：

**一个面向多镜头生成视频的、仅凭 video+caption 的全自动场景连续性诊断 benchmark。**

目标不是给模型一个笼统的 consistency score，而是像影视 continuity supervisor 一样，指出生成视频中具体哪里、哪个元素、在哪些镜头之间发生了穿帮。主要贡献包括：

### 3.1 新任务：caption-only、全自动的双线连续性诊断
已有评测大多关注主体一致性、单镜头 prompt-following，或依赖参考视频、逐元素标注和人工判分。我们把任务形式化为**只有 prompt 和生成视频、无人工介入**的多镜头连续性诊断，并明确区分两种能力：prompt-grounded continuity 与 intrinsic self-consistency。前者评估模型是否遵守显式创作约束，后者评估模型是否维护自己建立的场景世界。

### 3.2 从 prompt-defined entities 到 generated scene evidence
对 caption 中主动提到的主体和物体，我们借鉴 entity-centric benchmark 的做法：从 caption 提取候选实体，用 grounding、CLIP gate、crop embedding 和结构化 MLLM judge 进行定位与跨镜比较。对 prompt 未提到但模型自己生成的背景信息，我们不预先构造复杂标注，而是在每个生成视频中在线分析三类可检测证据：global state、salient objects 和 spatial layout。这样既覆盖显式剧本要求，也覆盖模型自发建立的场景记忆。

### 3.3 Hybrid evaluator：结构化 MLLM 判断 + 确定性聚合
我们不把整段视频丢给 MLLM 直接打总分，而是将评测分解为可审计模块：shot 对齐、caption entity extraction、grounding、embedding comparison、view comparability grouping、global state / object / layout comparison。MLLM 用于结构化语义判断和 finding 解释，例如判断一个 crop 是否符合实体描述、两个实体是否为同一对象、某个场景变化是否构成连续性错误；最终分数由固定规则聚合。这避免了纯 MLLM judge 的黑箱性，同时保留其语义理解能力。

### 3.4 Coverage × Correctness：抗空洞刷分的指标设计
连续性评估有一个天然陷阱：画面越空洞、可比较元素越少，越不容易出现错误。我们因此不只报告 consistency accuracy，还报告 coverage / richness：模型生成了多少可检查的 prompt-grounded entities 和 intrinsic scene evidence。空洞视频 coverage 低，丰富但混乱的视频 correctness 低，只有"既可检查、又稳定"的生成结果才表现好。

### 3.5 无人工的可信度验证：受控扰动标定
既然没有人当裁判，指标的可信度靠**构造已知答案**自证：对自洽片段程序化注入已知漂移（改色、删道具、换脸、平移家具），要求对应分数单调下降，并测其检出率。这把每个分数从"经验阈值"变成"被验证的判据"，也把"如何认证一个自动视频连续性指标"本身作为方法贡献。

## 4. 数据构造流程

真实影视视频提供多镜头结构和连续性压力，但不直接作为 ground truth。发布的是原创 episode 的 **prompt(caption) + 生成视频 + 自动诊断结果**，外加一个**独立的扰动验证集**（用于标定评测器，非 benchmark 主数据）。

整体流程：

真实叙事视频
→ 镜头结构与连续性压力点挖掘
→ 抽象改写为原创 episode 的 shot-level prompt
→ 输入生成模型得到多镜头视频
→ 自动提取 prompt-grounded entities 与 intrinsic scene evidence
→ 全自动诊断 continuity errors

### 4.1 从真实叙事视频中挖掘结构
真实影视剧、短片、广告中天然存在同场景多镜头叙事（远景建立、近景刻画、特写道具、正反打、插入、遮挡后重现、道具交接、回到远景）。我们挖掘的不是具体人物或画面，而是**多镜头组织方式和连续性压力点**。

### 4.2 抽象改写为原创 episode（只产出 prompt）
为避免版权问题，真实视频只作结构来源。我们替换具体人物、地点、物体和情节，但保留其连续性挑战，写成 shot-level prompt。我们不发布人工逐元素 continuity contract；prompt 中主动提到的主体列表由评测器自动抽取，prompt 之外的场景细节由生成结果本身提供证据。

### 4.3 生成并诊断
将 episode prompt 输入不同视频生成模型，收集生成视频，按 §7 全自动诊断。最终发布数据：
- 原创 episode 的 shot-level prompt；
- 各模型生成视频；
- 自动抽取的 prompt-grounded entity list（用于审计，可重现生成）；
- 自动诊断结果与分数；
- 评测代码与指标；
- 独立的扰动验证集（自动带标签，用于评测器标定）。

## 5. Episode 组织

每个 episode 是同一场景内的短多镜头叙事，通常 4–8 个镜头。每个 episode 包含两部分输入，以及一个评测时自动产生的审计产物：

### 5.1 Shot-level Prompt
逐镜描述内容、动作、视角、镜头类型与时长。它定义模型要生成什么，也是评测时做 shot 对齐和提取 prompt-grounded entities 的文本依据。

### 5.2 Evaluation Target（可选、仅供分析分组）
标注该 episode 主要制造的连续性压力（主体持续 / 道具重现 / 背景稳定 / 状态保持 / 空间关系 / 光影氛围），仅用于结果分层分析，不参与判分、不作为标准答案。

### 5.3 Auto-extracted Prompt Entity List（评测产物）
评测器从 shot-level prompt 中自动抽取主动提到的 characters、objects、locations、actions 和显式 relations。它不是人工标注，也不是 hidden answer，而是为了让 prompt-grounded continuity 的检查对象可审计、可复现。

## 6. 连续性错误类型

第一版定义五类核心 continuity errors。它们可以发生在 prompt-grounded track，也可以发生在 intrinsic track：

- **Missing**：应持续存在的元素在后续镜头中消失。
- **Appearance Drift**：元素外观无理由变化（颜色、形状、材质、服装、图案、身份漂移）。
- **State Drift**：元素状态不连续（已拿起的道具无理由回到桌上等）。
- **Spatial Drift**：元素位置或相对关系不合理变化（门、窗、桌、沙发、墙画等空间关系不自洽）。
- **Lighting / Atmosphere Drift**：光源方向、色温、亮度、时间氛围突变。

## 7. 评测方法（双线、分解式、可复现流水线）

评测器判断生成视频是否同时满足外部 prompt-grounded continuity 和内部 self-consistency。整体流程全部自动；视觉模块提供可复现证据，MLLM 提供结构化语义判断和诊断解释，最终分数由确定性规则聚合。（完整算法、参数与分数公式见配套 `pipeline_plan.md`，此处为方法概述。）

### 7.1 Stage 1 — Shot 检测与关键帧
经典内容差分法（HSV 直方图卡方距离 + 自适应阈值）切分镜头。实测发现 AI 生成视频常把分镜渲成**连续运镜**，纯差分对硬切可靠、对软分镜漏检、对快速运镜误检；因此采用 **prompt-anchored 对齐**：把检测峰吸附到 prompt 声明的分镜时间点。每个 shot 取"清晰度−运动"最优的中段帧为关键帧，避开运动模糊。

### 7.2 Stage 2 — Prompt-grounded entity extraction 与 grounding
从每镜 prompt 解析主动提到的实体及 referring expression，用 GroundingDINO / open-vocabulary detector 定位，并用 CLIP gate 与 crop quality gate 选择 canonical crop。对人物和物体计算 DINOv2 / 人脸 embedding；对动作可构造多帧 annotated grid。MLLM 以结构化 JSON 判断 crop 是否符合实体描述、动作是否被正确呈现。该阶段输出 prompt-grounded coverage 与后续跨镜比较所需的证据。

### 7.3 Stage 3 — Intrinsic scene evidence 与视角可比性
对 prompt 未显式指定但模型自己生成的场景信息，评测器不试图枚举所有背景细节，而只检查三类普适且可检测的证据：

- **Global State**：风格、光影、时间、天气、整体氛围；
- **Salient Objects**：显著背景物体和陈设；
- **Spatial Layout**：粗空间结构和相对关系。

为了避免正常机位变化造成误判，先进行 view / comparable-shot grouping：融合全图/背景 embedding、局部特征匹配、RANSAC 几何验证、大结构 anchor 分布；几何证据弱时可用 MLLM 判断可比性，但必须输出结构化原因。后续强比较只在可比视角组内进行。

### 7.4 Stage 4 — 跨镜头连续性比较
两条 track 共享"只在可检查机会中比较"的原则：

- **Prompt-grounded track**：对 caption 主动提到的实体比较 presence、identity、appearance、state、action 和显式 relations。DINOv2 / ArcFace / attribute comparison 给出可计算分数，MLLM 只输出结构化 fidelity / consistency judgment。
- **Intrinsic track**：在可比视角组内比较 global state 的分类与统计、salient objects 的存在/属性/位置、spatial layout 的粗关系图。缺乏可比证据时标记为 unverified，不直接判错。

### 7.5 Stage 5 — 汇总与 typed findings
把各分数汇总为分维度与总体分数，并输出元素级、类型化、可定位的 finding：`{element, error_type, affected_shots, evidence(分数与观测), confidence, severity}`。这正是 benchmark 的诊断价值所在。

## 8. 指标设计

### 8.1 四个主维度：Coverage × Correctness

我们优先报告四个主维度，而不是只报一个黑箱总分：

- **Prompt-grounded Coverage**：caption 主动提到的实体、动作和关系中，有多少被生成并可检查；
- **Prompt-grounded Consistency**：可检查的 prompt-grounded elements 中，有多少跨镜保持稳定；
- **Intrinsic Scene Richness / Coverage**：模型自己生成出多少可检测、可比较的 global state / salient objects / spatial layout evidence；
- **Intrinsic Self-Consistency**：这些 intrinsic scene evidence 在可比镜头中有多少保持稳定。

这四项共同避免两种错误激励：只生成空洞画面以逃避错误，或只遵守 prompt 中主体但背景世界不断漂移。

### 8.2 Scene Continuity Score（按机会数归一化的标量主指标）
需要单一排序时，用**机会数归一化**汇总，避免不同富度不可比：
```
SCS = Σ_c (w_c · score_c · opp_c) / Σ_c (w_c · opp_c)
opp_c = 该维度的可比较机会数（实体出现的 shot 对数、同视角组内帧对数等）
```
其中 `c` 覆盖 prompt-grounded 与 intrinsic 两条 track。分项包括 Subject / Face、Prompted Object、Action / State、Global State、Salient Object、Spatial Layout。我们同时报告 `opp_c` 或 coverage，避免模型通过减少可比较内容获得虚高 consistency。

### 8.3 诊断型指标
体现诊断能力：每 episode 的 continuity error 数、每类错误频率、每类元素错误率、不同镜头结构下的错误率、major error rate。用于揭示模型是易丢道具、易破坏空间关系，还是在特定镜头结构中更易失败。

### 8.4 MLLM 可靠性与受控扰动验证
MLLM 不直接产生最终总分，而是输出结构化判断与 findings。其可靠性通过两种方式约束：一是与非 MLLM evidence 一起进入确定性聚合，二是在受控扰动验证集上检查分数单调性与 finding 定位。扰动包括改色、删道具、换脸、平移家具、改变光影或布局；报告每类分数的检出率、误报率、定位准确率，以及阈值 τ 的 ROC 拐点。

## 9. 缓解方法（辅助实验）

建议包含一个简单 baseline：**Continuity Memory Prompting**——生成后续镜头时显式维护一个 scene continuity memory，记录前文需保持的元素、属性、状态、空间关系与光影，作为约束加入后续 prompt。其作用不是提出新模型，而是验证 benchmark 能否指导改进：哪些错误可被显式记忆减少、哪些仍难缓解、失败究竟来自记忆不足、空间建模不足还是细粒度视觉控制不足。预期它对主体和显著道具有帮助，但对复杂背景陈设、空间布局、光影连续提升有限——进一步说明本 benchmark 暴露的是当前模型尚未解决的深层问题。

## 10. 总结

真实影视中同一场景往往包含较长的多镜头叙事，天然要求人物、道具、背景、空间和光影在剪辑之间保持连续。随着视频生成走向多镜头叙事，仅评估主体一致性已不够。

我们提出一个**仅凭 video 与 caption、全自动**的多镜头场景连续性诊断 benchmark：从真实叙事视频挖掘多镜头连续性压力，抽象为原创 episode 的 shot-level prompt；沿 prompt-grounded continuity 与 intrinsic self-consistency 两条线诊断生成视频中的 continuity errors，输出元素级、类型化 findings；用 coverage × correctness 的指标设计抗空洞刷分，用受控扰动在无人工条件下标定指标可信度。

最终，本 benchmark 把多镜头视频评测从"主体是否一致"推进到"整个场景是否像真实影视一样，同时遵守创作者要求并维护自身建立的场景世界"。
