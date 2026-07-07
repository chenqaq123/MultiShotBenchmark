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

1. **Prompt-specified Continuity**：caption / shot prompt 中明确提到的人物、物体和地点，应在相关镜头中被正确生成并跨镜保持身份与外观稳定。它衡量模型能否遵守创作者明确给出的连续性约束。
2. **Model-emergent Self-Consistency**：prompt 未明确规定、但模型自己生成并在多个镜头中复现的实体（配角、未指定的道具家具、墙面陈设等背景元素），以及模型建立的整体场景空间，应在后续可比镜头中与自身保持自洽。它衡量模型能否维护自己生成出的场景世界。

这两者共同构成模型在影视创作中的连续性能力。前者对应剧本和分镜中明确要求的主体与关键道具；后者对应观众在观看多镜头场景时自然建立的角色、物体与空间记忆——模型自发引入的人物是否保持同一身份、自发生成的道具陈设是否稳定、同一视角下的背景结构是否自洽。

### 2.2 输入设定：video + caption，主流程自动

Benchmark 的输入是：

- 生成视频；
- episode / shot-level caption（或 structured script）；
- prompt-specified entity list（人物、物体、地点），采用结构化格式（含各实体的 scheduled shots），由 structured script 直接给出或人工标注（第一版不做自由文本 caption 的自动实体解析，以保持流水线确定性）。

**镜头边界不是输入**：评测器始终用双信号（HSV 直方图 + 降采样像素差分）加 cross-gap 验证的算法从视频本身检测 shot 边界，不依赖外部提供的切分——cross-gap 验证还能区分真切变与生成视频常见的分块生成接缝。我们不使用参考视频，也不依赖逐元素的人工 continuity contract。评测主流程由确定性视觉模块驱动、可自动运行；caption 只负责提供外部要求和镜头语义，模型自由生成出的额外实体与背景则由评测器从视频中在线检测、聚类和比较。评测器自身的质量（尤其是背景同视角分组）另用少量 per-shot 视角标签做验证，但这些标注不参与对生成模型的判分。

### 2.3 与普通 prompt-following 的区别

Prompt-specified continuity 不是简单的单镜头 prompt-following。它包含两个层次：

- **Shot realization / coverage**：caption 明确要求的主体或物体是否在应出现的镜头中被生成并可定位；
- **Cross-shot stability**：这些已生成的 prompt-specified elements 是否在多个镜头中保持身份和外观稳定。

因此，"prompt 中要求蓝杯子但第一镜没有杯子"是 coverage / realization failure；"第一镜生成了蓝杯子，后续镜头变成白杯子"是 continuity failure。两者都重要，但需要分开报告，避免把 prompt-following 和跨镜连续性混成一个黑箱分数。

## 3. Benchmark 定位与贡献

本 benchmark 定位为：

**一个面向多镜头生成视频的、以 video+caption 为主输入、对生成模型的诊断全自动的场景连续性 benchmark。**

目标不是给模型一个笼统的 consistency score，而是像影视 continuity supervisor 一样，指出生成视频中具体哪里、哪个元素、在哪些镜头之间发生了穿帮。主要贡献包括：

### 3.1 新任务：caption + video 的双线连续性诊断
已有评测大多关注主体一致性、单镜头 prompt-following，或依赖参考视频。我们把任务形式化为**以 prompt 和生成视频为主输入**的多镜头连续性诊断，并明确区分两种能力：prompt-specified continuity 与 model-emergent self-consistency。前者评估模型是否遵守显式创作约束，后者评估模型是否维护自己建立的场景世界（自发实体与背景空间）。

### 3.2 从 prompt-specified entities 到 model-emergent entities 与背景空间
对 caption 中主动提到的主体和物体，我们借鉴 entity-centric benchmark 的做法：从 caption 提取候选实体，用 open-vocabulary grounding、crop / face / object embedding 进行定位与跨镜比较。对 prompt 未提到但模型自己生成的元素，我们不预先构造复杂标注，而是在每个生成视频中在线做 open-world proposal 检测、跨镜聚类，得到 model-emergent tracks；同时对背景做前景剥离后的同视角分组，比较同视角下的背景一致性。这样既覆盖显式剧本要求，也覆盖模型自发建立的角色、物体与场景空间记忆。

### 3.3 分解式、可审计的确定性评测器
我们不把整段视频丢给一个模型直接打总分，而是将评测分解为可审计的确定性模块：keyframe 抽取、prompt entity 解析、open-world proposal 检测与分割、前景/背景分离、实体关联、人脸 / 物体 embedding 比较、背景同视角分组与一致性计算。每个模块都产出可复现的中间证据（crops、masks、embeddings、tracks、same-view groups），最终分数由固定规则从这些证据聚合。整条流水线避免黑箱总分：任何一个分数都能回溯到具体的检测、embedding 相似度或分组结果。

### 3.4 Coverage × Correctness：抗空洞刷分的指标设计
连续性评估有一个天然陷阱：画面越空洞、可比较元素越少，越不容易出现错误。我们因此不只报告 consistency，还报告 coverage / richness：模型生成了多少可检查的 prompt-specified entities、多少可复现的 model-emergent tracks，以及多少可比较的同视角背景对。空洞视频 coverage 低，丰富但混乱的视频 correctness 低，只有"既可检查、又稳定"的生成结果才表现好。

### 3.5 评测器可信度验证：同视角分组质量自检
评测中风险最高的一环是背景**同视角分组**——分组错误会直接污染背景一致性分数。我们因此把分组质量本身作为被验证对象：用少量人工 per-shot 视角标签（pair 级 same-view / different-view 真值由 shot 标签自动展开）计算 pairwise precision / recall / F1，并报告 over-merge 与 over-split 率和 view confusion matrix。这让"背景一致性分数是否可信"成为可度量、可报告的量，且这部分标注只用于评测器自检，不参与对生成模型的评分。

## 4. 数据构造流程

真实影视视频提供多镜头结构和连续性压力，但不直接作为 ground truth。发布的是原创 episode 的 **prompt(caption) + 生成视频 + 评测器检测的镜头边界（供审计） + 自动诊断结果**，外加用于评测器自检的**少量 per-shot 视角标签**（用于验证背景分组质量，非 benchmark 主数据）。

整体流程：

真实叙事视频
→ 镜头结构与连续性压力点挖掘
→ 抽象改写为原创 episode 的 shot-level prompt
→ 输入生成模型得到多镜头视频
→ 解析 prompt-specified entities、检测 model-emergent proposals、剥离前景/背景
→ 关联实体、人脸/物体 embedding 比较、背景同视角分组与一致性
→ 输出各线连续性分数与元素级明细

### 4.1 从真实叙事视频中挖掘结构
真实影视剧、短片、广告中天然存在同场景多镜头叙事（远景建立、近景刻画、特写道具、正反打、插入、遮挡后重现、道具交接、回到远景）。我们挖掘的不是具体人物或画面，而是**多镜头组织方式和连续性压力点**。

### 4.2 抽象改写为原创 episode（只产出 prompt）
为避免版权问题，真实视频只作结构来源。我们替换具体人物、地点、物体和情节，但保留其连续性挑战，写成 shot-level prompt。我们不发布人工逐元素 continuity contract；prompt 中主动提到的主体列表由评测器自动抽取，prompt 之外的场景细节由生成结果本身提供证据。

### 4.3 生成并诊断
将 episode prompt 输入不同视频生成模型，收集生成视频，按 §7 自动诊断。最终发布数据：
- 原创 episode 的 shot-level prompt；
- 各模型生成视频；
- prompt-specified entity list（用于审计，可重现生成）；
- 评测器检测的镜头边界（shots.json，供审计复现）；
- 自动诊断结果与分数（含 entity tracks、same-view groups、各维度指标与 failure-case 对照图）；
- 评测代码与指标；
- 用于同视角分组质量自检的少量人工 per-shot 视角标签。

## 5. Episode 组织

每个 episode 是同一场景内的短多镜头叙事，通常 4–8 个镜头。每个 episode 包含两部分输入，以及一个评测时自动产生的审计产物：

### 5.1 Shot-level Prompt
逐镜描述内容、动作、视角、镜头类型与时长。它定义模型要生成什么，也是评测时做 shot 对齐和解析 prompt-specified entities 的文本依据。

### 5.2 Evaluation Target（可选、仅供分析分组）
标注该 episode 主要制造的连续性压力（主体持续 / 道具重现 / 背景稳定 / 空间关系 / 自发实体复现），仅用于结果分层分析，不参与判分、不作为标准答案。

### 5.3 Prompt-specified Entity List（评测产物）
评测器从 shot-level prompt / structured script 中抽取主动提到的 characters、objects 和 locations（含各自的 scheduled shots）。它不是 hidden answer，而是为了让 prompt-specified continuity 的检查对象可审计、可复现；每个实体记录其应出现的镜头，用于判断 presence 与跨镜一致性。

## 6. 连续性漂移类型

第一版定义分数所刻画的连续性漂移维度（低分对应下述现象），按三条评测线组织。它们用于解释各维度分数的含义与结果分层，benchmark 只报告连续分数、不对是否"构成错误"下判定：

**Prompt-specified track**
- **Missing**：prompt 指定的人物 / 物体在其 scheduled shot 中缺失。
- **Identity Drift**：prompt 指定的人物在多镜之间身份漂移（人脸 embedding 不一致）。
- **Appearance Drift**：prompt 指定的物体外观无理由变化（颜色、形状、材质、图案）。
- **Background / Location Drift**：prompt 指定的地点或背景发生漂移。

**Model-emergent track**
- **Emergent Identity Instability**：模型自发生成的配角身份不稳定。
- **Emergent Object Drift**：模型自发生成的物体 / 道具反复变化或消失。
- **Track Fragmentation / Merge**：同一视觉元素被拆成多个 emergent track，或不同元素被错误合并为同一 track。

**Same-view background track**
- **Same-view Background Drift**：同一视角组内背景结构 / 布局 / 外观不自洽。
- **View Over-merge / Over-split**：不同视角被错误合并，或同一视角因生成细节漂移被错误拆分。

## 7. 评测方法（三线、分解式、可复现流水线）

评测器沿三条线判断生成视频：prompt-specified 实体一致性、model-emergent 自一致性、同视角背景一致性。整条流水线全部由确定性视觉模块驱动，每个阶段产出可复现证据，最终分数由固定规则聚合。（完整算法、参数与分数公式见配套 `pipeline_plan.md`，此处为方法概述。）

### 7.1 Stage 1 — Shot 边界检测与关键帧抽取
先由算法检测镜头边界（不依赖外部 shot list）：相邻帧的 HSV 直方图差分与降采样像素差分做鲁棒归一化（median+MAD）取候选切点，再做 cross-gap 验证——真切变的内容在边界后持续不同，而闪帧与分块生成接缝会在数帧内恢复。然后对每个 shot 在首/中/尾稳定位置抽取代表性关键帧，用 Laplacian 清晰度、亮度、对比度过滤转场帧、严重运动模糊和黑/白/fade 帧（背景可见比例过低的关键帧在同视角分组阶段排除）。

### 7.2 Stage 2 — Prompt-specified entity 解析与关联
从 structured script 得到 characters、objects、locations 及其 scheduled shots。每个实体的 description 作为独立 grounding 查询在关键帧中单独定位（open-vocabulary detector 返回 token 级标签，独立前向使检测框直接归属实体）；每个 scheduled shot 内按 grounding 分数与 crop embedding 连续性（与已选 appearance 质心的相似度）的加权分数贪心选出该实体的 appearance，抑制单帧误检。scheduled shot 中未匹配到的实体记为 missing，构成 prompt-specified coverage。

### 7.3 Stage 3 — Open-world proposal 与前景/背景分离
除 prompt 实体外，对每个关键帧做 open-world 检测 / 分割（GroundingDINO / YOLO / RT-DETR + SAM / Mask2Former），得到 person / face / object / 显著背景区域 proposals，并按面积、置信度、清晰度、重复度过滤。合并前景实体的 mask 得到 foreground mask，其补集为 background mask，并记录 background visible ratio，供背景特征的 masked pooling 使用。

### 7.4 Stage 4 — 实体关联与 model-emergent tracks
未被分配给任何 prompt 实体的 proposal 进入 model-emergent pool，通过 embedding 跨镜聚类形成 emergent tracks（要求至少在两个 shot / 多个关键帧出现、embedding 相似度足够高、类型一致）。这样区分"模型遵守 prompt 的实体"与"模型自发建立并需自洽的实体"。

### 7.5 Stage 5 — 人物与物体一致性
- **人物**：对 person crop 做人脸检测 / 对齐（SCRFD）、ArcFace embedding，计算 track 内 pairwise 与 centroid 人脸相似度；分别输出 prompt-specified 与 emergent 的 presence / detection / 身份一致性指标。
- **物体**：对 object crop 提取 DINOv2 crop embedding，小物体（面积比低于阈值）与 HSV 颜色直方图交集加权融合；计算 track 内 pairwise 与 centroid 相似度；分别输出 prompt-specified 与 emergent 的 presence / 外观一致性指标。

### 7.6 Stage 6 — 背景同视角分组
为避免正常机位变化造成误判，先做同视角分组：对 background mask 下的 DINOv2 patch 做 masked pooling 得背景特征，融合 depth layout 与 edge layout 相似度，按固定权重加权、episode 内 p5/p95 归一化为 same-view score，再用 mutual-kNN 图聚类成 same-view groups；建边额外要求原始背景相似度越过绝对下限，防止相对归一化在全异视角的 episode 中虚构分组。背景可见比例过低的帧不强行参与分组。（局部特征几何匹配在生成视频的纹理漂移下不可靠，第一版不使用，留作可选扩展。）

### 7.7 Stage 7 — 同视角背景一致性与分组质量
在每个 same-view group 内计算背景 embedding、depth、edge 的组内一致性，并按组内 pair 数加权汇总为 episode 级同视角一致性。同时用 view confusion matrix 与少量人工 per-shot 视角标签（pair 真值由 shot 标签自动展开）评估分组质量（pairwise P/R/F1、over-merge / over-split 率），作为背景一致性可信度的自检。

### 7.8 Stage 8 — 汇总与分数明细
把各线的 embedding 相似度、presence / recurrence、组内一致性等汇总为分维度分数（见 §8），并给出元素级、可定位的分数明细：`{element/track, comparison_type, shot_pairs, similarity_scores}`——每个 prompt 实体 / emergent track / same-view group 在各镜头对上的一致性分数，以及分数最低的镜头对。全 episode 分数最低的 K 个镜头对被物化为并排 crop 对照图（failure_cases/），作为可直接目检的证据。benchmark 只报告分数与证据，不做 pass/fail 判定。

## 8. 指标设计

### 8.1 四组指标：Coverage × Correctness，不混为单一总分

我们报告四组指标，而非一个黑箱总分：

- **Prompt-specified entity consistency**：prompt_character_presence_rate、prompt_face_mean / min / centroid_similarity、prompt_object_presence_rate、prompt_object_mean / min / centroid_similarity。衡量 prompt 要求的人 / 物是否出现且跨镜保持身份与外观。
- **Model-emergent self-consistency**：emergent_character_count / recurrence_rate / face_mean_similarity、emergent_object_count / recurrence_rate / object_mean_similarity，以及 identity / object fragmentation rate。衡量模型自发实体是否在多镜中自洽。
- **Background same-view consistency**：same_view_group_count、average_same_view_group_size、intra_group_bg / depth / edge similarity、episode_same_view_consistency。衡量同视角下背景是否稳定。
- **View grouping quality**：pairwise precision / recall / F1、over-merge / over-split 率、view confusion matrix。衡量分组本身是否可信。

其中 presence rate、count、recurrence rate、group size 等构成 coverage / richness，similarity / pass rate 构成 correctness。这样既避免"只生成空洞画面逃避错误"，也避免"只守住 prompt 主体而自发背景不断漂移"。

### 8.2 分数的分层报告
除总体分数外，按元素类型（人物 / 物体 / 背景）、按 track（prompt-specified / model-emergent）、按镜头结构分别报告一致性分数分布，并报告 fragmentation / merge 与 over-merge / over-split 等结构性统计。用于揭示模型是易丢道具、易破坏空间关系，还是在特定镜头结构中一致性更低——全部以连续分数与统计量呈现，不设错误判定阈值。

### 8.3 只报分数，不做检测判定
本 benchmark 输出连续分数（embedding 相似度、presence / recurrence rate、组内一致性等），可复现、可审计；它不把生成结果判为"有错 / 无错"——如何据分数排序或设线，留给使用者按需应用。指标中的 `*_pass_rate` 是阈值参数化的统计量（阈值随结果文件一并公布、可复算），刻画相似度分布越过给定水位的比例，不构成判定。评测中风险最高的同视角分组另用少量人工 per-shot 视角标签做 pairwise P/R/F1 与 over-merge / over-split 自检；缺乏可比证据（背景可见比例过低、无同视角对）的维度直接报为无可比机会（coverage=0），不计入一致性分数。

## 9. 缓解方法（辅助实验）

建议包含一个简单 baseline：**Continuity Memory Prompting**——生成后续镜头时显式维护一个 scene continuity memory，记录前文需保持的实体、属性与空间结构，作为约束加入后续 prompt。其作用不是提出新模型，而是验证 benchmark 能否指导改进：哪些错误可被显式记忆减少、哪些仍难缓解、失败究竟来自记忆不足、空间建模不足还是细粒度视觉控制不足。预期它对 prompt 指定的主体和显著道具有帮助，但对模型自发实体的自洽和复杂背景的同视角连续提升有限——进一步说明本 benchmark 暴露的是当前模型尚未解决的深层问题。

## 10. 总结

真实影视中同一场景往往包含较长的多镜头叙事，天然要求人物、道具、背景和空间在剪辑之间保持连续。随着视频生成走向多镜头叙事，仅评估主体一致性已不够。

我们提出一个**以 video 与 caption 为主输入**的多镜头场景连续性诊断 benchmark：从真实叙事视频挖掘多镜头连续性压力，抽象为原创 episode 的 shot-level prompt；用分解式、确定性的视觉流水线（检测、分割、人脸/物体 embedding、背景同视角分组），沿 prompt-specified continuity、model-emergent self-consistency、same-view background consistency 报告生成视频的连续性分数，并给出元素级、可定位的分数明细；用 coverage × correctness 的指标设计抗空洞刷分，用同视角分组质量自检约束背景一致性的可信度。整条流水线只输出连续分数、不设 pass/fail 阈值。

最终，本 benchmark 把多镜头视频评测从"主体是否一致"推进到"整个场景是否像真实影视一样，同时遵守创作者要求并维护自身建立的场景世界"。
