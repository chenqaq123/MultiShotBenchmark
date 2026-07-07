# 多镜头视频场景连续性评测流水线

本文档对应 `proposal.md` 的落地方案。目标是保持评测框架清晰：输入只有生成视频与 shot-level caption；评测分为 **Prompt-grounded Continuity** 和 **Intrinsic Self-Consistency** 两条线；MLLM 用于结构化语义判断和诊断解释，最终分数由确定性规则聚合。

---

## 0. 输入与输出

### 输入

- `video`: 一个 episode 的多镜头生成视频，或多个已切分 shot video。
- `captions`: episode / shot-level prompt，包含每个镜头的内容、动作、视角、镜头类型与时长。

### 输出

- `prompt_entities.json`: 从 caption 自动抽取的主动提及主体列表，用于审计。
- `shot_table.json`: shot 边界、关键帧、可比视角组。
- `scores.json`: coverage / consistency / SCS 等分数。
- `findings.json`: 元素级、类型化、可定位的 continuity findings。
- 可选 audit artifacts：canonical crops、关键帧、视角组可视化、MLLM JSON response。

---

## 1. 总体框架

```
video + shot captions
        |
Shot alignment and keyframes
        |
Caption entity extraction
        |
+---------------------------+     +-----------------------------+
| Prompt-grounded track     |     | Intrinsic scene track       |
| caption 主动提到的元素     |     | 模型自己建立的场景信息       |
+---------------------------+     +-----------------------------+
        |                                   |
Grounding / crop / embedding        View grouping / scene evidence
        |                                   |
Structured MLLM judgment            Global / object / layout comparison
        |                                   |
        +---------------+-------------------+
                        |
       deterministic aggregation + typed findings
```

两个 track 共享三个原则：

1. **先建立可检查机会，再判断错误**：没有足够视觉证据时标记为 `unverified`，不直接记为错误。
2. **分开报告 coverage 与 correctness**：避免空背景、低细节视频靠"无可检查内容"获得高分。
3. **MLLM 不直接给总分**：MLLM 输出结构化 JSON，最终分数由固定公式计算。

---

## 2. Stage 1 — Shot 对齐与关键帧

### 2.1 Shot 对齐

优先使用 caption 中的分镜结构。如果模型输出为多个 shot video，直接使用边界；如果输出为连续视频，则结合内容差分与 prompt-anchored alignment。

推荐实现：

```
1. HSV histogram / PySceneDetect 检测候选边界。
2. 根据 caption 声明的 shot 数、时长或顺序做最近邻 / DTW 对齐。
3. 对快速运镜导致的伪切进行最小镜头长度与光流平滑过滤。
```

AI 生成视频经常把分镜渲成连续运镜，所以纯 shot boundary detection 不可靠。prompt-anchored alignment 的作用不是提供连续性答案，而是保证后续评测按同一组 shot 进行。

### 2.2 关键帧采样

每个 shot 采样 3-6 帧，覆盖开头、中段、结尾。用于实体 grounding 的 canonical frame 由清晰度和运动幅度共同选择：

```
score(f) = sharpness(f) - lambda * motion(f)
sharpness = var(Laplacian(gray))
motion = mean |f_t - f_{t-1}| 或光流幅度
```

输出：

```json
{
  "shot_id": "S03",
  "time_range": [8.0, 11.0],
  "caption": "...",
  "keyframes": ["S03_f1.png", "S03_f2.png", "S03_f3.png"]
}
```

---

## 3. Stage 2 — Prompt-Grounded Continuity

这条线评估 caption 主动提到的主体、物体、地点、动作状态和显式关系。它学习 EntityBench 的经验，但目标从 entity consistency 扩展到 continuity diagnosis。

### 3.1 Caption Entity Extraction

从 shot-level caption 自动抽取：

- `characters`: 人物及稳定描述，如外观、服装、身份标签；
- `objects`: 关键道具和被交互物；
- `locations`: 明确地点或场景区域；
- `actions/states`: 拿起、放下、递交、坐下、站立等状态变化；
- `relations`: A beside B, object on table, person left of person 等显式关系。

可以用规则 + LLM 解析，但输出必须是结构化、可审计 JSON：

```json
{
  "shot_id": "S02",
  "characters": [
    {
      "id": "woman_1",
      "description": "woman in a beige cardigan",
      "expected_visibility": "visible"
    }
  ],
  "objects": [
    {
      "id": "mug_1",
      "description": "blue ceramic mug",
      "state": "on the table"
    }
  ],
  "relations": [
    {"subject": "mug_1", "relation": "on", "object": "table"}
  ]
}
```

这里的 entity list 来自 caption，不是人工 continuity contract。它只定义 prompt-grounded track 的候选对象。

### 3.2 Grounding 与 Canonical Evidence

对每个 prompt entity，在对应 shot 的 sampled frames 中定位：

- open-vocabulary detector：GroundingDINO / OWL-ViT；
- CLIP text-image similarity gate；
- crop quality gate：面积、清晰度、遮挡程度；
- 人物可加 face detector / ArcFace；
- 动作用 annotated multi-frame grid。

canonical crop 的选择可以采用：

```
selection = detector_confidence * clip_similarity * sharpness_score * area_score
```

输出每个 `(shot, entity)` 的三态：

- `present`: 检出且通过 CLIP / quality gate；
- `weak`: 有候选但置信度不足，可供 MLLM 审计；
- `absent`: 未检出或无可用证据。

### 3.3 Prompt-Grounded 分数

**Coverage**

```
PG-Coverage = # present prompt entity slots / # scheduled prompt entity slots
```

按 character / object / location / action 分开报告。

**Entity Fidelity**

对 canonical crop 与 caption description 做结构化 MLLM 判断，输出 1-10 并归一化到 `[0,1]`。criteria 可采用类型化 rubric：

- character: face / hair / clothing / build
- object: shape / color_texture / proportions / details
- location: layout / color_mood / landmarks / perspective

**Cross-shot Consistency**

对跨多个 shot 出现的 prompt entity：

```
emb_{e,i} = DINOv2(crop_{e,i})
centroid_e = normalize(mean_i emb_{e,i})
CS_emb(e) = mean_i cos(emb_{e,i}, centroid_e)
```

人物身份可另算：

```
CS_face(p) = mean_{i<j} cos(ArcFace(face_{p,i}), ArcFace(face_{p,j}))
```

MLLM 可做 anchor-vs-each 或 set-based structured judge，输出：

```json
{
  "is_same": true,
  "similarity": 8,
  "criteria": {"clothing": 7, "face": 9},
  "reason": "..."
}
```

最终分数使用固定聚合，而不是 MLLM 总分。

**Prompt-Grounded Consistency**

```
PG-Consistency =
  weighted_mean(CS_emb, CS_face, state_consistency, relation_consistency)
```

只在可检查机会中计算；低 fidelity 的错误 crop 可 gate out，避免"错误但一致"获得高分。

---

## 4. Stage 3 — Intrinsic Self-Consistency

这条线评估模型自己建立的场景世界。我们不为每个 case 提前构造复杂 registry，也不试图追踪所有背景细节。每个生成视频在线处理，只检查三类普适且可检测的证据。

### 4.1 可检测范围

**Global State**

- 风格：真人、动漫、3D、插画、油画感等；
- 光影：暖光/冷光、明亮/昏暗、侧光/逆光；
- 时间与天气：白天、夜晚、雨、雪、雾；
- 整体色调和氛围。

**Salient Objects**

- 显著背景物和陈设：门、窗、沙发、桌子、灯、墙画、植物、书架、招牌、车辆等；
- 不包含 prompt 主动提到的主体和关键道具，它们归 prompt-grounded track。

**Spatial Layout**

- 粗空间结构：门/窗/桌/沙发/墙面/道路/建筑的位置；
- 粗关系：left/right/above/near/on/behind/in-front-of；
- 不评精确几何距离、纹理细节、书架上几本书这类脆弱细节。

### 4.2 View / Comparable-Shot Grouping

Intrinsic track 的关键是先判断哪些镜头可比较。不要在人物特写中检查远景里的窗户，也不要把正反打的正常视角变化误判为空间漂移。

推荐融合四类信号：

```
S_view(i,j) =
  w1 * background_embedding_similarity
+ w2 * local_feature_geometry_score
+ w3 * large_anchor_layout_similarity
+ w4 * global_state_similarity
```

可用模块：

- 背景 DINOv2 / CLIP embedding；
- 人物 mask 后的局部特征匹配：SIFT / SuperPoint / LightGlue；
- RANSAC homography / fundamental matrix inlier ratio；
- 大结构 anchor 的 bbox 分布；
- MLLM checkability judgment 作为低置信补充。

输出 viewpoint groups：

```json
{
  "group_id": "vp2",
  "shots": ["S02", "S04", "S06"],
  "confidence": 0.82,
  "evidence": ["shared sofa/window layout", "high bg embedding", "RANSAC inliers"]
}
```

### 4.3 组内一致性分数

**Global State Consistency**

对每个 shot 提取：

- full-frame / background embedding；
- brightness / saturation / color temperature / dominant color；
- CLIP zero-shot 或 MLLM 分类：day/night, warm/cool, rainy/foggy, style type。

```
S_global(G) = mean_{i<j in G} sim(global_state_i, global_state_j)
```

**Salient Object Consistency**

在同一视角组内，自动提取显著背景物候选，并只保留面积大、清晰、可命名、可重复观察的对象。比较方式：

- detector / segmenter 定位候选；
- crop embedding 相似度；
- 属性一致性：颜色、形状、材质、图案；
- 位置一致性：归一化 bbox 与局部上下文。

```
S_object(G) = mean over verifiable salient object matches
```

如果某个物体在 reference shot 中出现，但后续 shot 不覆盖相同区域，则记为 `unverified`，不扣分。只有同一局部场景可见且该物体无证据存在时，才报告 Missing。

**Spatial Layout Consistency**

把每个可比 shot 转为粗 layout graph：

```json
{
  "nodes": ["sofa", "table", "window", "lamp"],
  "edges": [
    ["table", "in_front_of", "sofa"],
    ["lamp", "right_of", "table"],
    ["window", "behind", "sofa"]
  ]
}
```

比较同一视角组内可检查边的一致率：

```
S_layout(G) =
  # consistent checkable relations / # checkable relations
```

只比较两个 anchor 都清楚可见的关系。

### 4.4 Intrinsic 分数

```
IS-Coverage =
  # verifiable intrinsic evidence / # potential salient evidence

IS-Consistency =
  weighted_mean(S_global, S_object, S_layout)
```

`IS-Coverage` 不应鼓励无限挖小细节，因此只统计通过 salience / clarity / repeatability 过滤的 evidence。主文可报告 `Scene Richness` 作为可解释版本。

---

## 5. MLLM 的角色

MLLM 不是唯一检测器，也不是最终打分器。它只承担三类任务：

1. **结构化语义判断**：entity fidelity、action fidelity、same/different、scene checkability。
2. **疑难视角判断**：当几何和 embedding 证据不足时，判断两个镜头是否可比较。
3. **finding explanation**：把数值证据转成可读诊断。

所有 MLLM 调用必须满足：

- 输入有限、局部、可审计，例如 crops、frame grid、paired frames；
- 输出 JSON schema；
- 1-10 分归一化到 `[0,1]`；
- 最终 score 由 evaluator 固定公式聚合；
- 保存 response 供抽样检查。

示例 schema：

```json
{
  "verdict": "consistent",
  "score": 8,
  "criteria": {"color": 8, "shape": 9, "position": 7},
  "evidence": "The lamp remains on the right side of the table.",
  "confidence": 0.78
}
```

---

## 6. 聚合指标

### 6.1 四个主维度

```
PG-Coverage
PG-Consistency
IS-Coverage / Scene Richness
IS-Consistency
```

这四项应作为主要 leaderboard 表格和散点图展示。

### 6.2 Scene Continuity Score

当需要单一排序时：

```
SCS = sum_c w_c * score_c * opp_c / sum_c w_c * opp_c
```

其中 `c` 包括：

- prompt character / face；
- prompt object；
- prompt action / state；
- prompt relation；
- intrinsic global state；
- intrinsic salient object；
- intrinsic spatial layout。

`opp_c` 是可比较机会数，例如：

- 同一实体出现的 shot 对数；
- 同一视角组内 frame / shot 对数；
- 可检查 layout relation 数；
- 可检查 salient object matches。

同时报告 coverage，避免 `SCS` 掩盖空洞生成。

### 6.3 Typed Findings

输出格式：

```json
{
  "track": "intrinsic",
  "element": "lamp_around_table",
  "error_type": "SpatialDrift",
  "affected_shots": ["S01", "S04"],
  "evidence": {
    "layout_relation": "lamp right_of table -> lamp left_of table",
    "score": 0.42,
    "threshold": 0.65
  },
  "confidence": 0.81,
  "severity": "major"
}
```

错误类型沿用：

- Missing
- Appearance Drift
- State Drift
- Spatial Drift
- Lighting / Atmosphere Drift

---

## 7. 受控扰动验证

为验证 evaluator 可信度，构造独立扰动集。对原本较稳定的视频片段注入可控错误：

- 改色：杯子、衣服、灯光色温；
- 删除/添加：显著道具或背景物；
- 替换：人脸、服装、墙画；
- 平移：家具、窗户、灯；
- 改变全局状态：夜晚变白天、暖光变冷光。

验证指标：

```
Detection Rate: 注入错误被对应 finding 检出的比例
False Positive Rate: 未扰动区域被误报的比例
Localization Accuracy: affected shots 是否命中
Monotonicity: 扰动强度增加时 score 是否单调下降
```

扰动验证不能证明覆盖所有生成模型的自然错误，但可以证明每个指标对其目标错误类型有基本灵敏度。

---

## 8. 实现模块清单

| 阶段 | 工具 / 模型 | 作用 |
|---|---|---|
| Shot 对齐 | PySceneDetect / HSV diff / DTW | shot boundary 与 prompt 对齐 |
| Caption 解析 | rules + LLM JSON parser | prompt entity list |
| Entity grounding | GroundingDINO / OWL-ViT | prompt entities 与 salient objects 定位 |
| Text-image gate | CLIP | 检测结果过滤 |
| Embedding | DINOv2 / CLIP image | crop / frame 相似度 |
| Face identity | InsightFace / ArcFace | 人物身份一致 |
| View grouping | DINO bg embedding + SIFT/SuperPoint/LightGlue + RANSAC | 可比视角分组 |
| Layout | bbox relation graph | 粗空间关系 |
| Global state | color stats + CLIP zero-shot + MLLM | 光影、时间、风格、氛围 |
| MLLM | structured JSON judge | 语义判断与 findings |

---

## 附录 A：当前穿帮样例如何落入通用框架

当前 `data/穿帮镜头/1.mp4 + 2.mp4` 可作为 pipeline sanity check，而不是 benchmark 定义本身。

可见问题：

1. **Prompt-grounded / subject appearance drift**：主体 3 的上衣在圆领针织与开衫之间漂移。
2. **Intrinsic / salient object state drift**：桌面道具数量在可比餐桌视角中变化。
3. **Intrinsic / spatial layout negative control**：卡座、百叶窗、挂画大体稳定，应该得到较高背景结构分，避免 evaluator 全部误报。

本 case 也暴露了 Stage 1 的现实问题：AI 视频常把分镜渲成连续运镜，纯内容差分会漏检软分镜或误检快速运镜，因此正式 pipeline 应采用 shot detector + prompt-anchored alignment，而不是只依赖无监督切分。
