# Pipeline Plan: Multi-shot 生成视频中的实体一致性与同视角背景分组

## 1. 目标

本 pipeline 面向 multi-shot 生成视频，核心目标是评估视频中实体和背景在多个 shot 之间的一致性。

需要区分两类一致性对象：

1. **Prompt-specified entities**  
   prompt 中明确指定的主体、人物、物体、地点或背景元素。

2. **Model-emergent entities**  
   prompt 中没有明确指定，但模型自己生成并在多个 shot 中重复出现的实体、物体、背景结构或视觉细节。

二者需要在同一套流程中处理，但指标上需要区分，不能混为一个总分。

---

## 2. 输入与输出

### 2.1 输入

```text
Input:
- multi-shot generated video
- structured prompt-specified entity list, if available
```

shot 边界不作为输入：评测器始终从视频本身用算法检测 shot 边界（见 Step 1），
避免评测依赖外部提供的切分。prompt-specified entity list 采用 §5.3 的结构化
格式，可由人工标注或 structured prompt 直接给出（v1 不做自由文本 caption 的
自动实体解析，以保持流水线确定性）。

### 2.2 输出

```text
Output:
- shot-level keyframes
- foreground / background masks
- character / object / background crops or regions
- prompt-specified entity consistency metrics
- model-emergent self-consistency metrics
- same-view background groups
- same-view background consistency metrics
- lowest-consistency pair exhibits (failure cases)
```

---

## 3. 总体流程

```text
Multi-shot generated video
→ shot boundary detection
→ shot-level keyframe extraction
→ prompt-specified entity parsing
→ open-world proposal extraction
→ foreground / background separation
→ entity association

→ character consistency evaluation
→ object consistency evaluation
→ background same-view grouping
→ same-view background consistency evaluation
→ final metrics and failure-case evidence
```

这里的关键是：  
prompt 指定实体和模型自发实体共享检测、分割、特征提取和聚类模块，但在指标汇总时分开统计。

---

## 4. Step 1: Shot Boundary Detection and Keyframe Extraction

### 4.0 Shot 边界检测（算法自主，不依赖外部输入）

对每对相邻帧计算两路差分信号：

```text
d_hist[t] = 1 - correlation(HSV histogram_t, HSV histogram_{t+1})   # 色调/光照变化
d_pix[t]  = mean |gray_t - gray_{t+1}|  (降采样灰度)                # 构图/布局变化
```

两路信号各自做鲁棒归一化（median + MAD），候选切点须满足：

```text
- robust z-score >= shot_adaptive_k（默认 8 个 MAD）
- 是 ±2 帧窗口内的局部极大值
- 至少一路原始差分越过绝对下限（shot_min_diff / shot_min_pix_diff）
```

候选切点再做 **cross-gap 验证**：比较切点前 gap 帧与后 gap 帧（默认 gap=3）。
真切变的内容在边界之后持续不同（跨间隙差分 ≈ 单步差分）；闪帧和生成视频的
分块生成接缝会在数帧内恢复（跨间隙差分坍缩回运动基线）。验证只在越过绝对
下限的信号通道上进行：

```text
accept iff 存在已触发通道 c:
    cross_diff_c >= max(0.5 * step_diff_c, floor_c)
```

最短 shot 长度约束（默认 0.5s）防止过碎切分。

### 4.1 关键帧抽取目的

从每个 shot 中抽取代表性关键帧，用于后续实体检测、特征提取、背景分组和一致性计算。

### 4.2 做法

每个 shot 在 15% / 50% / 85% 位置各取一帧（first-stable / middle / last-stable），
在每个位置的小窗口内选 Laplacian 清晰度最高的帧；shot 过短时只取中间帧。
若全部候选被质量过滤拒绝，回退到 shot 中间帧，保证每个 shot 至少一个关键帧。

### 4.3 低质量帧过滤

过滤以下帧：

```text
- 转场帧
- 严重 motion blur
- 黑屏 / 白屏 / fade frame
```

过滤信号：

```text
- Laplacian blur score（下限）
- brightness（上下限，滤黑/白屏）
- contrast / gray std（下限，滤平坦帧）
```

背景可见比例不在此处过滤（mask 在关键帧之后才产生）；背景可见比例过低的
关键帧在 same-view 分组阶段被排除（§12.2）。

---

## 5. Step 2: Prompt-specified Entity Parsing

### 5.1 目的

从 prompt 或 structured script 中得到明确需要评估的实体。

### 5.2 实体类型

```text
- character / person
- object
- background / location
```

### 5.3 结构化表示

```json
{
  "characters": [
    {
      "id": "char_01",
      "description": "the main character specified in the prompt",
      "scheduled_shots": [1, 2, 4]
    }
  ],
  "objects": [
    {
      "id": "obj_01",
      "description": "the object specified in the prompt",
      "scheduled_shots": [2, 3]
    }
  ],
  "backgrounds": [
    {
      "id": "bg_01",
      "description": "the location or background specified in the prompt",
      "scheduled_shots": [1, 2, 3, 4]
    }
  ]
}
```

### 5.4 作用

这一部分只负责建立 prompt-specified entity list。  
它用于后续判断：

```text
- prompt 指定的实体是否出现
- prompt 指定的实体是否跨 shot 保持一致
```

---

## 6. Step 3: Open-world Proposal Extraction

### 6.1 目的

除了 prompt 指定实体，还需要发现模型自己生成的实体和视觉元素。

这些元素不来自 prompt，但可能在多个 shot 中重复出现，例如：

```text
- 模型自己生成的配角
- prompt 未指定的家具
- prompt 未指定的道具
- prompt 未指定的墙面装饰
- prompt 未指定的背景结构
- 模型自己生成的服饰细节
```

### 6.2 方法

对每个 keyframe 提取 open-world proposals：

```text
keyframe
→ open-world detection / segmentation
→ proposal masks
→ proposal crops
→ proposal embeddings
```

可用模块：

```text
- GroundingDINO, when text prompts are available
- YOLO / RT-DETR, for generic person / object detection
- SAM / SAM2, for mask proposal extraction
- Mask2Former, for panoptic segmentation
```

### 6.3 Proposal 过滤

过滤掉以下 proposal：

```text
- 面积过小
- 面积过大且无明确结构
- confidence 过低
- 严重模糊
- 与其他 proposal 高度重复
- 纯背景纹理碎片
```

输出保留：

```text
- person proposals
- face proposals
- object proposals
- salient background-region proposals
```

---

## 7. Step 4: Foreground / Background Separation

### 7.1 目的

背景同视角分组需要尽量减少人物、物体和主体动作的干扰，因此需要获得 foreground mask 和 background mask。

### 7.2 流程

```text
keyframe
→ detect prompt-specified entities
→ detect open-world foreground proposals
→ segment foreground
→ merge foreground masks
→ background mask = not foreground mask
```

### 7.3 输出

对每个 keyframe 输出：

```text
- character masks
- object masks
- foreground mask
- background mask
- background-visible ratio
```

### 7.4 注意

不要求对被遮挡区域做 inpainting。  
后续提取背景特征时，可以直接使用 background mask 对 DINOv2 patch tokens 做 masked pooling。

---

## 8. Step 5: Entity Association

### 8.1 目的

将每个 keyframe 中检测到的 proposal 关联到两类实体集合：

```text
A. Prompt-specified entities
B. Model-emergent entities
```

### 8.2 Prompt-specified association

每个 prompt 实体的 description 作为**独立的 grounding 查询**单独前向
（GroundingDINO 返回 token 级标签而非完整查询短语，标签解析不可靠；独立
前向使检测框直接归属实体）。

每个 scheduled shot 内的候选按以下分数贪心选出一个 appearance：

```text
score(p) = (1 - w) * grounding_score(p)
         + w * max(0, cosine(dino_emb(p), centroid(已选 appearances 的 dino_emb)))
```

首个 shot 只用 grounding score；后续 shot 通过 embedding 连续性抑制
grounding 误检（w = assoc_continuity_weight，默认 0.5）。

如果某个 scheduled shot 中没有匹配到 prompt 指定实体，记录为 missing。
非 scheduled shot 中的额外出现也被记录（标记 scheduled=false）。

### 8.3 Model-emergent association

对于没有被分配到 prompt-specified entity 的 proposal，进入 model-emergent pool。

```text
unassigned proposals
→ embedding extraction
→ cross-shot clustering
→ emergent entity tracks
```

model-emergent track 的要求是：

```text
- 至少在两个 shot 或多个 keyframes 中出现
- embedding similarity 足够高
- proposal 类型一致或视觉特征一致
```

---

## 9. Step 6: Character Consistency Evaluation

### 9.1 适用对象

```text
- prompt-specified characters
- model-emergent characters
```

### 9.2 流程

```text
character crop / person proposal
→ face detection
→ face alignment
→ ArcFace / InsightFace embedding
→ face similarity computation
```

### 9.3 推荐模型

```text
- RetinaFace / SCRFD for face detection
- ArcFace / InsightFace for face embedding
```

### 9.4 指标定义

设同一 character track 中第 i 次出现的人脸 embedding 为 `f_i`。

#### Pairwise FaceSim

```text
FaceSim(i, j) = cosine(f_i, f_j)
```

#### Centroid FaceSim

```text
face_centroid = mean(normalized(f_i))
FaceSim_centroid(i) = cosine(f_i, face_centroid)
```

### 9.5 Prompt-specified character metrics

```text
- prompt_character_presence_rate
- prompt_face_detection_rate
- prompt_face_mean_similarity
- prompt_face_min_similarity
- prompt_face_similarity_std
- prompt_face_centroid_similarity
- prompt_identity_pass_rate
```

`*_pass_rate` 是阈值参数化的统计量（阈值随 metrics.json 公布，可复算），
不构成对生成结果的 pass/fail 判定。

### 9.6 Model-emergent character metrics

```text
- emergent_character_count
- emergent_character_recurrence_rate
- emergent_face_mean_similarity
- emergent_face_min_similarity
- emergent_face_similarity_std
- emergent_face_centroid_similarity
- emergent_identity_fragmentation_rate
```

### 9.7 解释

Prompt-specified character metrics 评价：

```text
prompt 要求的人是否出现，并保持同一身份
```

Model-emergent character metrics 评价：

```text
模型自己生成的人物是否在多 shot 中自一致
```

---

## 10. Step 7: Object Consistency Evaluation

### 10.1 适用对象

```text
- prompt-specified objects
- model-emergent objects
```

### 10.2 流程

```text
object proposal / crop
→ DINOv2 feature extraction
→ object embedding similarity
→ object consistency metrics
```

### 10.3 特征

物体相似度以 DINOv2 crop-level embedding 为主；小物体的 crop 结构信息少，
额外融合 HSV 颜色直方图交集：

```text
ObjectSim(i, j) = cosine(dino_i, dino_j)

当 area_ratio_i 与 area_ratio_j 均 < small_object_area_ratio 时：
ObjectSim(i, j) = (1 - w) * cosine(dino_i, dino_j)
               + w * histogram_intersection(hist_i, hist_j)
（w = color_hist_weight，默认 0.3）
```

该融合相似度同时用于 emergent object 聚类和所有 object 一致性指标。

### 10.4 Centroid 相似度

与人脸一致，object track 同样计算 centroid 相似度：

```text
centroid = mean(normalized(dino_i))
ObjectSim_centroid(i) = cosine(dino_i, centroid)
```

### 10.5 Prompt-specified object metrics

```text
- prompt_object_presence_rate
- prompt_object_mean_similarity
- prompt_object_min_similarity
- prompt_object_similarity_std
- prompt_object_centroid_similarity
- prompt_object_pass_rate
```

### 10.6 Model-emergent object metrics

```text
- emergent_object_count
- emergent_object_recurrence_rate
- emergent_object_mean_similarity
- emergent_object_min_similarity
- emergent_object_similarity_std
- emergent_object_centroid_similarity
- emergent_object_fragmentation_rate
```

### 10.7 解释

Prompt-specified object metrics 评价：

```text
prompt 要求的物体是否出现，并保持外观一致
```

Model-emergent object metrics 评价：

```text
模型自己生成的物体或道具是否在多 shot 中保持一致
```

---

## 11. Step 8: Background Same-view Grouping

### 11.1 目的

将 keyframes 按同一视角分组。

这里的“同一视角”指：

```text
- 相机朝向相近
- 背景结构位置相近
- 主要空间布局相近
- 构图相近
```

不是简单的同一地点。

例如：

```text
同一个厨房的正面视角和侧面视角不应被分到同一 same-view group。
```

---

## 12. Step 8.1: Background Feature Extraction

### 12.1 DINOv2 background patch feature

对每个 keyframe 提取 DINOv2 patch tokens：

```text
tokens_i = DINOv2(keyframe_i)
```

使用 background mask 选择背景 patch：

```text
bg_tokens_i = tokens_i[background_mask_i == 1]
```

做 masked pooling：

```text
bg_feat_i = mean(bg_tokens_i)
bg_feat_i = normalize(bg_feat_i)
```

背景相似度：

```text
S_dino_bg(i, j) = cosine(bg_feat_i, bg_feat_j)
```

### 12.2 背景可见性

记录每帧背景可见比例：

```text
background_visible_ratio_i = area(background_mask_i) / area(frame_i)
```

如果背景可见比例太低，该帧不应强行参与 same-view grouping。

---

## 13. Step 8.2: Layout Feature Extraction

为了避免将同一地点的不同视角误合并，需要加入 layout 特征。

### 13.1 Depth layout

```text
depth_i = DepthModel(keyframe_i)
depth_layout_i = resize(depth_i, 64x64)
S_depth(i, j) = similarity(depth_layout_i, depth_layout_j)
```

similarity 可以使用：

```text
- cosine similarity
- SSIM
```

### 13.2 Edge layout

```text
edge_i = EdgeDetector(keyframe_i)
edge_layout_i = resize(edge_i, 64x64)
S_edge(i, j) = similarity(edge_layout_i, edge_layout_j)
```

EdgeDetector 可以是：

```text
- Canny
- HED
- DexiNed
```

### 13.3 可选 segmentation layout

如果已有 panoptic 或 segmentation 结果，可以得到区域级布局：

```text
S_seg(i, j) = similarity(seg_layout_i, seg_layout_j)
```

---

## 14. Step 8.3: Same-view Score

### 14.1 综合分数

```text
same_view_score(i, j) =
    0.60 * S_dino_bg(i, j)
  + 0.25 * S_depth(i, j)
  + 0.15 * S_edge(i, j)
```

v1 不使用几何匹配信号：生成视频的局部纹理漂移使 SuperPoint/LoFTR 类局部
特征匹配不可靠，收益不足以抵消其脆弱性。若未来引入，可作为第四路弱信号
（权重 ≤0.10）加入，其余权重等比缩减。

### 14.2 归一化

所有分量在 episode 内归一化到 `[0, 1]`：

```text
S_norm = clip((S - p5) / (p95 - p5), 0, 1)
```

p5/p95 是 episode 内的相对归一化：当所有视角都不同时，最相似的 pair 也会被
拉到高分。因此建边时额外要求原始 `S_dino_bg` 越过绝对下限
（same_view_raw_dino_floor，默认 0.55），防止相对归一化无中生有地制造分组。

---

## 15. Step 8.4: Same-view Grouping

### 15.1 建图

```text
node = keyframe
edge(i, j) exists if same_view_score(i, j) > threshold
edge_weight = same_view_score(i, j)
```

推荐使用 mutual-kNN graph：

```text
edge(i, j) exists if:
    i in topK(j) and j in topK(i)
```

### 15.2 聚类

```text
mutual-kNN graph
→ graph clustering
→ same-view groups
```

可用：

```text
- connected components
- Louvain / Leiden
- HDBSCAN
- agglomerative clustering
```

### 15.3 每组代表帧

对每个 same-view group 选择 medoid frame，兼顾组内代表性与画面质量：

```text
candidates = { i : mean_j same_view_score(i, j) >= medoid_tolerance * best }
medoid = argmax_{i in candidates} (background_visible_ratio_i, blur_score_i)
```

即在平均相似度接近最优（默认 98%）的帧中，优先选背景可见比例高的，
再以清晰度决胜。

---

## 16. Step 9: Background Consistency Evaluation

### 16.1 组内背景一致性

对每个 same-view group `G`：

```text
intra_group_bg_similarity(G)
    = mean_{i,j in G, i<j} S_dino_bg(i, j)
```

### 16.2 组内 layout 一致性

```text
intra_group_depth_similarity(G)
    = mean_{i,j in G, i<j} S_depth(i, j)

intra_group_edge_similarity(G)
    = mean_{i,j in G, i<j} S_edge(i, j)
```

### 16.3 组内综合一致性

```text
intra_group_same_view_score(G)
    = mean_{i,j in G, i<j} same_view_score(i, j)
```

### 16.4 Episode-level aggregation

```text
episode_same_view_consistency
    = weighted_mean_G intra_group_same_view_score(G)
```

权重可以使用组内 pair 数：

```text
weight_G = |G| * (|G| - 1) / 2
```

---

## 17. Final Metric Report

最终报告按**元素类型**组织为三大块（人物 / 物体 / 背景），不混合为单一总分，
全部为连续分数与统计量。人物与物体两块内部各分 prompt-specified 与
model-emergent 两条 track；背景块不做实体级拆分，报告同视角一致性。

```text
metrics.json
├── characters
│   ├── prompt_specified      # prompt 指定人物
│   └── model_emergent        # 模型自发人物
├── objects
│   ├── prompt_specified      # prompt 指定物体
│   └── model_emergent        # 模型自发物体
└── background                # 同视角背景一致性
```

### 17.1 Characters

prompt_specified：

```text
- prompt_character_presence_rate
- prompt_face_detection_rate
- prompt_face_mean / min / centroid_similarity, std
- prompt_identity_pass_rate
- coverage: n_entities, n_comparable_pairs
- per_entity 明细（含逐 pair 分数与最低分镜头对）
```

model_emergent：

```text
- emergent_character_count
- emergent_character_recurrence_rate
- emergent_face_mean / min / centroid_similarity, std
- emergent_identity_fragmentation_rate
- coverage: cluster_total, no_face_tracks
- per_track 明细
```

### 17.2 Objects

prompt_specified：

```text
- prompt_object_presence_rate
- prompt_object_mean / min / centroid_similarity, std
- prompt_object_pass_rate
- coverage: n_entities, n_comparable_pairs
- per_entity 明细
```

model_emergent：

```text
- emergent_object_count
- emergent_object_recurrence_rate
- emergent_object_mean / min / centroid_similarity, std
- emergent_object_fragmentation_rate
- coverage: cluster_total
- per_track 明细
```

### 17.3 Background

```text
- same_view_group_count / average_same_view_group_size
- intra_group_bg / depth / edge_similarity
- episode_same_view_consistency
- coverage: n_grouped / n_excluded keyframes, n_comparable_pairs
- per_group 明细（含 medoid）
```

---

## 18. Failure Case Taxonomy

### 18.1 Prompt-specified entity failures

```text
- prompt 指定人物缺失
- prompt 指定人物身份漂移
- prompt 指定物体缺失
- prompt 指定物体外观漂移
- prompt 指定背景或地点发生漂移
```

### 18.2 Model-emergent self-consistency failures

```text
- 模型自发生成的配角身份不稳定
- 模型自发生成的物体反复变化
- 模型自发生成的背景装饰物消失或改变
- 同一视觉元素被拆成多个 emergent tracks
- 不同视觉元素被错误合并为同一个 emergent track
```

### 18.3 Same-view grouping failures

```text
- 同一地点不同视角被错误合并
- 同一视角因为生成细节漂移被错误拆分
- 前景人物遮挡导致背景特征失真
- 局部纹理幻觉导致几何匹配误判
```

---

## 19. Final Pipeline Summary

```text
1. Detect shot boundaries (dual-signal + cross-gap verification)
   and extract shot-level keyframes.

2. Parse prompt-specified entities (structured list):
   - characters
   - objects
   - backgrounds / locations

3. Extract open-world proposals:
   - per-entity grounding passes + generic open-world vocabulary
   - persons / faces / objects / salient background regions
   - SAM masks per proposal

4. Segment foreground and background:
   - foreground mask
   - background mask
   - background visible ratio

5. Associate proposals:
   - per scheduled shot, pick by grounding score + embedding continuity
   - cluster unassigned recurring proposals as model-emergent entities

6. Evaluate character consistency:
   - ArcFace / InsightFace embeddings
   - pairwise + centroid similarity
   - prompt-specified character metrics
   - emergent character self-consistency metrics

7. Evaluate object consistency:
   - DINOv2 object embeddings (+ color-histogram blend for small objects)
   - pairwise + centroid similarity
   - prompt-specified object metrics
   - emergent object self-consistency metrics

8. Group backgrounds by same-view:
   - DINOv2 background patch pooling
   - depth layout similarity
   - edge layout similarity
   - mutual-kNN graph clustering with absolute background-similarity floor

9. Evaluate same-view background consistency:
   - intra-group background similarity
   - intra-group layout similarity
   - episode-level same-view consistency

10. Output final report and evidence:
   - characters / objects: prompt-specified and model-emergent consistency
   - background same-view consistency
   - lowest-consistency pair exhibits (failure_cases/)
```

---

## 20. Deliverables

```text
pipeline_plan.md

src/
  common.py                        # config defaults + shared utilities
  models.py                        # lazy singletons for pretrained models
  detect_shots.py
  extract_keyframes.py
  parse_prompt_entities.py
  extract_open_world_proposals.py
  segment_foreground_background.py
  associate_entities.py
  extract_face_features.py
  extract_object_features.py
  extract_background_features.py
  compute_same_view_score.py
  cluster_same_view.py
  evaluate_metrics.py
  export_failure_cases.py
  run_pipeline.py                  # end-to-end orchestrator (CLI)

scripts/
  run_eval.sh                      # env wrapper

configs/
  episode_*.json                   # per-episode: video / entities

outputs/<episode_id>/
  shots.json
  keyframes.json  keyframes/
  masks/                           # fg/bg PNG + per-proposal mask npz
  crops/
  embeddings/                      # proposals.npz, background.npz
  proposals.json
  entity_tracks.json
  same_view_groups.json
  metrics.json
  failure_cases/                   # lowest-consistency pair exhibits + manifest
```
