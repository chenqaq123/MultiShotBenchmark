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
- shot boundary / shot list
- prompt or structured script
- prompt-specified entity list, if available
```

其中 prompt-specified entity list 可以来自人工标注、structured prompt，或从 prompt 中解析得到。

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
- view confusion matrix
```

---

## 3. 总体流程

```text
Multi-shot generated video
→ shot-level keyframe extraction
→ prompt-specified entity parsing
→ open-world proposal extraction
→ foreground / background separation
→ entity association

→ character consistency evaluation
→ object consistency evaluation
→ background same-view grouping
→ same-view background consistency evaluation
→ final metrics and confusion analysis
```

这里的关键是：  
prompt 指定实体和模型自发实体共享检测、分割、特征提取和聚类模块，但在指标汇总时分开统计。

---

## 4. Step 1: Shot-level Keyframe Extraction

### 4.1 目的

从每个 shot 中抽取代表性关键帧，用于后续实体检测、特征提取、背景分组和一致性计算。

### 4.2 推荐做法

每个 shot 至少抽取一个中间关键帧：

```text
keyframe_s = middle_frame(shot_s)
```

如果需要增强鲁棒性，可以从每个 shot 抽取：

```text
- first stable frame
- middle frame
- last stable frame
```

### 4.3 低质量帧过滤

过滤以下帧：

```text
- 转场帧
- 严重 motion blur
- 黑屏 / 白屏 / fade frame
- 主体或背景几乎不可见的帧
```

可用过滤信号：

```text
- Laplacian blur score
- brightness / contrast
- foreground visible ratio
- background visible ratio
```

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

对于 prompt 中指定的实体：

```text
prompt entity description
+ keyframe proposal
→ matching score
→ assigned prompt entity id
```

匹配可以基于：

```text
- detection class / text grounding score
- crop embedding similarity
- spatial and temporal continuity
- face embedding, for characters
- object embedding, for objects
```

如果某个 scheduled shot 中没有匹配到 prompt 指定实体，记录为 missing。

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
- prompt_identity_pass_rate
```

### 9.6 Model-emergent character metrics

```text
- emergent_character_count
- emergent_character_recurrence_rate
- emergent_face_mean_similarity
- emergent_face_min_similarity
- emergent_face_similarity_std
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

推荐使用：

```text
- DINOv2 crop-level embedding
- DINOv2 patch-token embedding
```

对于小物体，可以辅助使用：

```text
- color histogram
- shape descriptor
- mask area ratio
```

### 10.4 指标定义

设同一 object track 中第 i 次出现的 embedding 为 `o_i`。

```text
ObjectSim(i, j) = cosine(o_i, o_j)
```

### 10.5 Prompt-specified object metrics

```text
- prompt_object_presence_rate
- prompt_object_mean_similarity
- prompt_object_min_similarity
- prompt_object_similarity_std
- prompt_object_pass_rate
```

### 10.6 Model-emergent object metrics

```text
- emergent_object_count
- emergent_object_recurrence_rate
- emergent_object_mean_similarity
- emergent_object_min_similarity
- emergent_object_similarity_std
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

## 14. Step 8.3: Weak Geometry Signal

### 14.1 作用

在生成视频中，局部纹理可能漂移，因此几何匹配只作为弱信号，不作为 hard rule。

### 14.2 流程

```text
background-only regions
→ local feature extraction
→ feature matching
→ RANSAC
→ geometry score
```

可用：

```text
- SuperPoint + LightGlue
- LoFTR
- ALIKED + LightGlue
```

### 14.3 几何信号

```text
- num_inliers
- inlier_ratio
- mean_reprojection_error
- matched_area_coverage
- homography_overlap
```

定义：

```text
S_geo(i, j) =
    clip(inlier_ratio, 0, 1)
    * clip(matched_area_coverage, 0, 1)
    * exp(-mean_reprojection_error / sigma)
```

---

## 15. Step 8.4: Same-view Score

### 15.1 综合分数

```text
same_view_score(i, j) =
    0.55 * S_dino_bg(i, j)
  + 0.20 * S_depth(i, j)
  + 0.15 * S_edge(i, j)
  + 0.10 * S_geo(i, j)
```

如果不使用几何信号：

```text
same_view_score(i, j) =
    0.60 * S_dino_bg(i, j)
  + 0.25 * S_depth(i, j)
  + 0.15 * S_edge(i, j)
```

### 15.2 归一化

所有分量统一归一化到 `[0, 1]`：

```text
S_norm = clip((S - p5) / (p95 - p5), 0, 1)
```

---

## 16. Step 8.5: Same-view Grouping

### 16.1 建图

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

### 16.2 聚类

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

### 16.3 每组代表帧

对每个 same-view group 选择 medoid frame：

```text
medoid = argmax_i mean_j same_view_score(i, j)
```

代表帧应满足：

```text
- 背景可见比例高
- 清晰度高
- foreground 遮挡少
- 与组内其他帧平均相似度高
```

---

## 17. Step 9: Background Consistency Evaluation

### 17.1 组内背景一致性

对每个 same-view group `G`：

```text
intra_group_bg_similarity(G)
    = mean_{i,j in G, i<j} S_dino_bg(i, j)
```

### 17.2 组内 layout 一致性

```text
intra_group_depth_similarity(G)
    = mean_{i,j in G, i<j} S_depth(i, j)

intra_group_edge_similarity(G)
    = mean_{i,j in G, i<j} S_edge(i, j)
```

### 17.3 组内综合一致性

```text
intra_group_same_view_score(G)
    = mean_{i,j in G, i<j} same_view_score(i, j)
```

### 17.4 Episode-level aggregation

```text
episode_same_view_consistency
    = weighted_mean_G intra_group_same_view_score(G)
```

权重可以使用组内 pair 数：

```text
weight_G = |G| * (|G| - 1) / 2
```

---

## 18. Step 10: View Confusion Matrix

### 18.1 目的

分析 same-view grouping 是否发生：

```text
- over-merge: 不同视角被合并
- over-split: 同一视角被拆开
```

### 18.2 有视角标签时

构建 confusion matrix：

```text
rows = predicted same-view groups
columns = ground-truth view labels
value = number of keyframes
```

### 18.3 无视角标签时

可以基于少量人工 pair annotation：

```text
pair_label(i, j) ∈ {same-view, different-view}
```

报告：

```text
- pairwise precision
- pairwise recall
- pairwise F1
- over-merge rate
- over-split rate
```

---

## 19. Final Metric Report

最终报告分为四组，不混合为单一总分。

### 19.1 Prompt-specified entity consistency

```text
- prompt_character_presence_rate
- prompt_face_mean_similarity
- prompt_identity_pass_rate
- prompt_object_presence_rate
- prompt_object_mean_similarity
- prompt_object_pass_rate
```

### 19.2 Model-emergent self-consistency

```text
- emergent_character_count
- emergent_character_recurrence_rate
- emergent_face_mean_similarity
- emergent_object_count
- emergent_object_recurrence_rate
- emergent_object_mean_similarity
```

### 19.3 Background same-view consistency

```text
- same_view_group_count
- average_same_view_group_size
- intra_group_bg_similarity
- intra_group_depth_similarity
- intra_group_edge_similarity
- episode_same_view_consistency
```

### 19.4 View grouping quality

```text
- pairwise precision
- pairwise recall
- pairwise F1
- over-merge rate
- over-split rate
- view confusion matrix
```

---

## 20. Failure Case Taxonomy

### 20.1 Prompt-specified entity failures

```text
- prompt 指定人物缺失
- prompt 指定人物身份漂移
- prompt 指定物体缺失
- prompt 指定物体外观漂移
- prompt 指定背景或地点发生漂移
```

### 20.2 Model-emergent self-consistency failures

```text
- 模型自发生成的配角身份不稳定
- 模型自发生成的物体反复变化
- 模型自发生成的背景装饰物消失或改变
- 同一视觉元素被拆成多个 emergent tracks
- 不同视觉元素被错误合并为同一个 emergent track
```

### 20.3 Same-view grouping failures

```text
- 同一地点不同视角被错误合并
- 同一视角因为生成细节漂移被错误拆分
- 前景人物遮挡导致背景特征失真
- 局部纹理幻觉导致几何匹配误判
```

---

## 21. Final Pipeline Summary

```text
1. Extract shot-level keyframes.

2. Parse prompt-specified entities:
   - characters
   - objects
   - backgrounds / locations

3. Extract open-world proposals:
   - persons
   - faces
   - objects
   - salient background regions

4. Segment foreground and background:
   - foreground mask
   - background mask
   - background visible ratio

5. Associate proposals:
   - assign matched proposals to prompt-specified entities
   - cluster unassigned recurring proposals as model-emergent entities

6. Evaluate character consistency:
   - ArcFace / InsightFace embeddings
   - prompt-specified character metrics
   - emergent character self-consistency metrics

7. Evaluate object consistency:
   - DINOv2 object embeddings
   - prompt-specified object metrics
   - emergent object self-consistency metrics

8. Group backgrounds by same-view:
   - DINOv2 background patch pooling
   - depth layout similarity
   - edge layout similarity
   - optional weak geometry signal
   - mutual-kNN graph clustering

9. Evaluate same-view background consistency:
   - intra-group background similarity
   - intra-group layout similarity
   - episode-level same-view consistency

10. Output final report:
   - prompt-specified entity consistency
   - model-emergent self-consistency
   - background same-view consistency
   - view confusion matrix
```

---

## 22. Deliverables

```text
pipeline_plan.md

src/
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

outputs/
  keyframes/
  masks/
  crops/
  embeddings/
  entity_tracks.json
  same_view_groups.json
  metrics.json
  view_confusion_matrix.png
  failure_cases/
```
