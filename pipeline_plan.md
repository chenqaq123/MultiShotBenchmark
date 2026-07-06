# 多镜头视频「场景连续性 / 穿帮」评测流水线 —— 落地方案

> 本方案针对 `data/穿帮镜头/1.mp4 + 2.mp4` 拼接成的单 case，给出**每个阶段的算法、参数、可迁移实现**，以及**每个分数的含义与计算公式**。
> Stage 1 已在本地(Mac, 仅 ffmpeg+numpy)实测验证，结果见文末附录 A。Stage 2/3 需 GPU，给出可直接迁移的模型与伪代码。

---

## 0. 记号与数据约定

- **Case**：`combined.mp4 = concat(1.mp4@24fps, 2.mp4@24fps)`，1280×720，20.08s，482 帧。
- **Prompt 真值分镜**（拼接时间轴，用于对齐/评估 Stage 1）：
  | shot | 时间 | 内容 | 视角类型 |
  |---|---|---|---|
  | S1 | 0–1s | 中景轨道横移 | wide/medium 正面 |
  | S2 | 1–8s | 越肩(主体1肩) | over-shoulder |
  | S3 | 8–11s | 近景 主体2/主体3 | close 正面 |
  | S4 | 11–17s | 越肩(主体1肩) | over-shoulder |
  | S5 | 17–20s | 近景 主体2/主体3 | close 正面 |
- **实体定义**（referring expression，供 GroundingDINO 使用）：
  | 实体 | prompt 描述 | GroundingDINO 文本 | 空间先验(线稿) |
  |---|---|---|---|
  | 场景基准 | 餐厅卡座 | — (背景) | 木质卡座+百叶窗+挂画 |
  | 主体1 | 深色连帽卫衣/清瘦苍白男 | `dark jacket thin man` | 单独一侧，背对/左肩 |
  | 主体2 | 黑色短袖微胖男 | `chubby man in black t-shirt` | 卡座右位 |
  | 主体3 | 米色针织开衫干净女 | `woman in beige cardigan` | 卡座左位 |

> ⚠️ 注意 prompt 与画面已有 fidelity 偏差（主体1 实际是夹克非连帽卫衣；主体3 上衣在不同 shot 在"圆领针织↔开衫"间漂移）。referring expression 用**稳定属性 + 座位先验**，不要照抄 prompt 里可能没兑现的细节。

---

## 1. Stage 1 — Shot 检测 + 关键帧提取

### 1.1 算法（经典，无需 GPU）
**内容差分法 (Content-based)**：逐帧算 HSV 直方图，相邻帧取卡方距离，自适应阈值找边界。

```
对每帧 t:
  H_t = HSV 三通道联合直方图 (8×8×8=512 bins)，L1 归一化
差分:  d_t = χ²(H_t, H_{t-1}) = 0.5 Σ (H_t - H_{t-1})² / (H_t + H_{t-1} + ε)
阈值:  thr = mean(d) + k·std(d)   (k=3，自适应)
边界:  d_t > thr 的 t；并做 min_scene_len(≥12帧=0.5s) 去抖 + 峰值 NMS
```

**服务器等价实现**：PySceneDetect `ContentDetector`（同为 HSV 加权差分）
```python
from scenedetect import detect, ContentDetector
scenes = detect("combined.mp4", ContentDetector(threshold=27, min_scene_len=12))
```

### 1.2 关键难点与对策（本 case 实测暴露）
AI 生成视频把"分镜"渲成**连续运镜**，只有**生成段之间**是硬切。纯差分法：
- 强峰 t=11.04s(段间硬切) ✅、t=7.88s(S2→S3) ✅；
- 弱/漏：1s、17s 切点；伪峰：2.21s(快速运镜)。

**对策（三选一，推荐 C）**：
- A. 降阈值 → 召回↑但伪切↑；
- B. 加运动通道：光流幅度突变辅助判切（区分"运镜渐变"vs"硬切"）；
- **C. Prompt-anchored 对齐（推荐）**：已知每个 case 的 prompt 声明了分镜数与时间，用**检测峰值去对齐/吸附到 prompt 声明的时间点**（DTW 或最近邻匹配），把"无监督切分"变成"有先验的边界修正"。评测本就要按 prompt 分镜比对，这一步天然合理。

### 1.3 关键帧选择（改进版，不要直接取中点）
中点帧常有运动模糊（如 S2 主体2 咀嚼）。在 shot 中段窗口 `[0.4L, 0.6L]` 内选：
```
score(f) = α·sharpness(f) − β·motion(f)
  sharpness = var(Laplacian(gray))         # 越大越清晰
  motion    = mean|f − f_prev| (光流或帧差)  # 越小越稳
取 argmax 的帧为关键帧
```
可每 shot 取 1 主 + 2 辅关键帧，增强后续检测鲁棒性。

### 1.4 输出
`shots.json`: `[{shot_id, t_start, t_end, keyframe_path, prompt_text, viewpoint_hint}]`

---

## 2. Stage 2 — 主体文本抽取 + GroundingDINO 检测（保留背景）

### 2.1 从 prompt 抽主体描述
每个 shot 的 prompt 文本 → 解析"本镜应出现哪些实体"以及其 referring expression：
- 规则/LLM 解析出现的 `<主体k>` 标签 → 查 §0 实体表得文本；
- 区分**应可见** vs **仅提及**（如"越肩"镜头主体1只有肩膀）→ 标 `visibility_expected ∈ {full, partial, absent}`，供 Missing 判定用。

### 2.2 GroundingDINO 开放词表检测
```python
# groundingdino (SwinT-OGC) 或 GroundingDINO-1.5
boxes, logits, phrases = predict(
    model, image=keyframe,
    caption="dark jacket thin man . chubby man in black t-shirt . woman in beige cardigan .",
    box_threshold=0.35, text_threshold=0.25)
```
**消歧（关键）**：两男易混。用**座位空间先验**约束：
- 主体1 → 取最靠画面左/背对镜头的框；主体3 → 卡座左位女性框；主体2 → 卡座右位框；
- 每实体保留 top-1 框（按 score×先验匹配度）；冲突时用匈牙利算法在"实体↔框"间做二分匹配。

### 2.3 主体裁剪 + 背景保留
- **主体 crop**：按框裁剪（外扩 10%），存 `crops/{shot}_{entity}.png`；
- **背景图**：把所有主体框 + 动态前景(食物盘子)区域 mask 掉，得到 `bg/{shot}.png`（结构评分用）。食物盘子可用同一 GroundingDINO 文本 `plate of food . bottle .` 检出后并入 mask。

### 2.4 输出
`detections.json`: 每 keyframe `{entity: {box, score, crop_path}}`, `bg_path`, `prop_boxes`。

---

## 3. Stage 3 — 相似度打分

> 核心思想：**主体走 embedding+人脸，背景走"空间视角分组→组内比较"**。所有跨 shot 比较都带**视角门控**（同视角强比，跨视角保守）。

### 3.1 主体外观自一致 —— DINOv2 embedding
```
对每个实体 e 在其出现的每个 shot i： v_{e,i} = DINOv2-ViT-L/14(crop_{e,i})  # CLS + mean-patch, L2 归一化
S_app(e) = mean_{i<j, gate(i,j)} cos(v_{e,i}, v_{e,j})
```
- **视角门控 gate(i,j)**：只在"两 shot 都能看到该主体的可比视角"时计入（如两个正面近景）；越肩镜头里主体1只有背影 → 不参与其外观比较，或单独作"背影一致"弱分。
- **含义**：主体的服装/体型/发型跨镜是否自洽。**抓 Appearance Drift**（如主体3 圆领↔开衫漂移）。

### 3.2 人脸身份一致 —— InsightFace (RetinaFace+ArcFace)
```
对每个人物 p 每 shot： f_{p,i} = ArcFace(face_crop)  # 512-d
S_face(p) = mean_{i<j} cos(f_{p,i}, f_{p,j})    # 无脸(背对)的 shot 跳过
```
- **含义**：身份是否被保持（无换脸/面部漂移）。ArcFace 对光照/角度比 DINO 更鲁棒，是人物一致性的主指标。
- 主体外观分数 `S_subject(p) = w1·S_app + w2·S_face`（人物 w2 大，非人实体只有 S_app）。

### 3.3 背景 / 场景一致 —— 空间视角分组 + 组内多分数

**Step A. 空间视角分组（无 MLLM，几何法）**
```
1. 每 keyframe 背景图 → 全局特征 g_i (DINOv2 on bg) ；
2. 局部特征匹配几何验证：SuperPoint+LightGlue(或 SIFT) 提点，
   两两匹配 → RANSAC 估单应 H_ij，记 inlier 数 n_ij；
3. 亲和矩阵 A_ij = 归一化(n_ij) 融合 cos(g_i,g_j)；
4. 谱聚类 / 阈值连通分量 → 视角组 {vp1(宽景), vp2(越肩), vp3(近景)...}；
   本质矩阵恢复相对位姿可进一步区分"同侧 vs 正反打"。
```
> 生成视频无真实 3D，几何法在纹理丰富的宽景(S1/S4背景)可靠、在暗糊近景弱 → 弱处退回全局 embedding 分组，并记 `group_confidence`。

**Step B. 组内分数**（同视角组内，背景已 mask 掉人和食物）

| 分数 | 计算 | 含义 / 抓什么 |
|---|---|---|
| `B_struct(g)` 结构一致 | 组内 keyframe 两两背景区域 mean **SSIM**（或 1−LPIPS），先用 H_ij 对齐 | 陈设/墙面/百叶窗/挂画是否稳定 → **背景漂移** |
| `B_geo(g)` 几何一致 | 组内 RANSAC 中位 **inlier_ratio**（或重投影误差倒数） | 场景是否扭曲/结构错乱 → **Spatial Drift / 结构崩** |
| `B_prop(g)` 道具一致 | 匹配道具框(瓶/盘)：`1 − |ΔN|/N_max` × 位置 IoU | **道具凭空增减**(本 case: shot1→shot4 盘子变多) |
| `L_seat` 座位一致(全局) | 各主体框中心的**左右次序 & 相对位置**与线稿一致率 | 三人座位**位置跳变** |

---

## 4. Stage 4 — 汇总与 findings

### 4.1 分数字典（case 级）
| 符号 | 名称 | 范围 | 归一化后含义 |
|---|---|---|---|
| S_pres(e) | 出现率 | [0,1] | 应出现且被检出的 shot 比例 → Missing |
| S_face(p) | 人脸一致 | [0,1] | 身份保持 |
| S_app(e) | 外观一致 | [0,1] | 服装/外观自洽 |
| B_struct/geo/prop | 背景三项 | [0,1] | 场景陈设/几何/道具自洽 |
| L_seat | 座位一致 | [0,1] | 空间关系自洽 |

**Scene Continuity Score**（主指标，按"机会数"归一化，避免空场景占便宜）：
```
SCS = Σ_c (w_c · score_c · opp_c) / Σ_c (w_c · opp_c)
  opp_c = 该维度的"可比较机会数"(如实体出现的 shot 对数、组内帧对数)
```
配合**诊断表**：每类 error 的发生率、每个实体/每个视角组的错误率。

### 4.2 Typed findings（诊断输出）
```json
{"entity":"主体3","type":"AppearanceDrift","shots":[2,4],
 "evidence":"S_app=0.71<τ; 圆领针织→开衫","confidence":0.8,"severity":"minor"}
{"entity":"table_props","type":"StateDrift/PropAdded","shots":[1,4],
 "evidence":"B_prop=0.6; 盘数 1→3","confidence":0.75,"severity":"major"}
```

---

## 5. 运行位置与模型清单

| 阶段 | 模型/工具 | 本地(Mac) | 服务器(GPU) |
|---|---|---|---|
| Shot 检测/关键帧 | ffmpeg + numpy / PySceneDetect | ✅ 已跑通 | ✅ |
| 主体检测 | GroundingDINO SwinT-OGC | ✗ | ✅ |
| 主体 embedding | DINOv2 ViT-L/14 | ✗(慢) | ✅ |
| 人脸 | InsightFace buffalo_l (RetinaFace+ArcFace) | ✗ | ✅ |
| 视角分组 | SuperPoint+LightGlue / OpenCV SIFT+RANSAC | ⚠️CPU可跑慢 | ✅ |
| 结构相似 | SSIM(skimage) / LPIPS | SSIM✅ LPIPS✗ | ✅ |

安装（服务器）：
```
pip install scenedetect opencv-python torch torchvision \
    groundingdino-py insightface onnxruntime-gpu \
    scikit-image lpips kornia   # kornia 提供 LoFTR/LightGlue
```

---

## 附录 A：Stage 1 本地实测结果（已验证）

- combined.mp4：482 帧 / 20.08s / 24fps。
- HSV χ² 差分：mean=0.0040, std=0.0117, max=0.1338；自适应阈值 mu+3sd=0.039。
- 检出边界(帧, 时间, 分数)：
  ```
  frame 189  t=7.88s   χ²=0.094   → 命中 S2→S3 (真值8s)
  frame 265  t=11.04s  χ²=0.134   → 命中 段间硬切 (真值11s) [最强]
  frame 438  t=18.25s  χ²=0.076   → 对应 S4→S5 (真值17s，偏晚)
  frame  53  t=2.21s   χ²=0.115   → 伪切(快速运镜)  ← 需运动通道/prompt对齐剔除
  ```
- 结论：纯差分对**硬切可靠、软分镜漏检、快速运镜误检** → 采用 §1.2-C Prompt-anchored 对齐。
- 关键帧样本已生成于 `scratchpad/keyframes/kf_shot{1..5}_*.png`（5 个 prompt 分镜中点帧）。

## 附录 B：本 case 肉眼可见的连续性问题（供验证 pipeline 召回）
1. **道具增减**：shot1(菜单+少量) → shot4(多盘咖喱+馒头碗) 桌面道具凭空增多 → `B_prop` 应低。
2. **主体3 上衣漂移**：shot2/3(圆领针织感) → shot4/5(明显开衫+白衬衫) → `S_app(主体3)` 应低。
3. **背景固定元素**（百叶窗/挂画/卡座）基本稳定 → `B_struct` 应高（作为负样本校验，避免全判错）。
