# 技术方案详细说明

## 一、整体方案：两阶段检测 + 坐标映射

我们没有用一个大模型端到端地做，而是拆成了三个独立模块串联：

```
输入图片
  │
  ├─ 第一阶段：YOLOv8n ──→ 定位符号位置（bounding box）
  │
  ├─ 第二阶段：轻量 CNN ──→ 对每个检测框做精细分类
  │
  └─ 第三阶段：坐标提取 ──→ 像素位置 → 真实数值坐标
```

这样拆的原因是：YOLO 擅长定位但分类精度一般（8 类符号很多长得像），CNN 专门做分类可以更准。两个模型各司其职。

## 二、每一步具体用了什么

### 第一步：符号定位 — YOLOv8n（nano 版）

- 模型：Ultralytics YOLOv8n，约 **3M 参数**
- 用的是最小的 nano 版，因为我们的机器只有 16GB 内存
- 输入：整张图片（640×640 resize）
- 输出：所有检测到的符号的 bounding box + 置信度
- 训练数据：我们自己用 matplotlib 生成的合成散点图，v1 版 200 张，v2 版 200 张（加了扫描噪声模拟）
- 训练参数：batch=8, epochs=30-50, CPU 训练
- 性能：mAP50 = 0.949

### 第二步：符号分类 — 自定义轻量 CNN

- 模型：自己写的 3 层卷积网络，约 **94K 参数**
- 结构：`Conv(1→16, 3×3) → Conv(16→32, 3×3) → Conv(32→64, 3×3) → FC → 9类`
- 输入：从 YOLO 检测框中心裁出的 **32×32 灰度 patch**
- 输出：9 类分类结果 — 8 种符号（filled/open × circle/square/triangle/diamond）+ background
- 分类决策逻辑：
  - 如果 CNN 和 YOLO 类别一致 → 直接采用
  - 如果 CNN 说是 background → 信任 YOLO（说明 CNN 没看清）
  - 如果 CNN 置信度 > 0.8 且和 YOLO 不一致 → 信任 CNN
  - 其他情况 → 信任 YOLO

### 第三步（可选）：SAHI 切片检测

- 目的：弥补小符号的漏检
- 做法：用 SAHI 框架把图片切成 320×320 的重叠小块，每块单独跑 YOLO，再把结果拼回来
- 然后和标准检测的结果合并，**用 10 像素半径去重**（两个检测中心距离 < 10px 视为重复）

### 第四步：后处理过滤

依次做了四层过滤：
1. **绘图区域过滤**：只保留落在数据区域内的检测，排除图例、轴标签、页边距的误检
2. **轴线近邻过滤**：去掉离坐标轴太近的检测（通常是刻度线被误检）
3. **置信度过滤**：去掉低置信度检测（阈值 0.20）
4. **小系列过滤**：一个类别如果只检测到 1-3 个点，大概率是误检，直接丢弃

### 第五步：坐标提取

- 需要预先知道坐标轴的刻度位置（像素坐标 + 对应的真实值）
- 用 `numpy.interp` 做线性插值，把每个符号的像素位置映射到真实数值
- 对数轴的处理：先在 log 空间做插值，再转回真实值
- 自动检测是否为对数轴：如果最大值/最小值 > 100，判定为对数轴

### 第六步：RANSAC 异常点过滤

- 用 scikit-learn 的 RANSACRegressor
- 在 log-x 空间做 2 阶多项式拟合（因为剂量-反应曲线通常是 S 型的）
- 自适应阈值：y 值范围的 15%
- 不符合拟合曲线的点被标记为 outlier 并剔除

## 三、API 串接方式

整个流水线是纯 Python 代码串联的，没有用外部 API 或云服务。关键函数调用链：

```python
# 1. 加载两个模型
detector, classifier = load_models(yolo_path, classifier_path)

# 2. 标准检测：YOLO 定位 → CNN 分类
std_dets = detect_symbols(image_path, detector, classifier, conf_threshold=0.30)

# 3. SAHI 检测：切片推理 → CNN 分类
sahi_dets = detect_symbols_sahi(image_path, yolo_path, classifier, conf_threshold=0.25)

# 4. 合并去重
merged = merge_and_dedup(std_dets, sahi_dets, dedup_radius=10)

# 5. 过滤（绘图区域、轴线、置信度）
filtered = filter_by_area(merged, plot_area)
filtered = filter_near_axes(filtered, plot_area)
filtered = filter_by_confidence(filtered, min_conf=0.20)

# 6. 坐标映射：像素 → 真实值
coords = extract_coordinates(filtered, x_axis, y_axis, plot_area)

# 7. 按符号类型分组
series = group_by_series(coords)

# 8. RANSAC 过滤异常点
for cls, points in series.items():
    inliers, outliers = ransac_filter_improved(points)
```

每个函数都在 `src/` 目录下，模型权重在 `models/` 目录下，不依赖任何外部服务，全部本地运行。

## 四、训练数据的生成方式

因为没有现成的标注数据，我们用代码自动生成合成训练图：

- 用 matplotlib 绘制随机散点图/折线图
- 随机选择符号类型、数量、位置、轴范围
- 自动生成 COCO 格式的标注（bounding box + 类别）
- v2 版本额外加入了**扫描伪影模拟**：高斯噪声、对比度降低、轻微模糊，让合成图更接近专利扫描件的风格

## 五、当前的局限性

| 环节 | 现状 | 原因 |
|------|------|------|
| 真实图召回率约 33% | 只提取到 16/~48 个数据点 | 合成数据和真实扫描件之间的域差距 |
| 缺失一条系列 | open_circle 系列完全没检出 | 空心符号在扫描件中和背景噪声难以区分 |
| 坐标轴需要手动配置 | 每张图要人工填入刻度像素位置 | 轴线 OCR 和自动刻度检测还没做 |
| 只验证了一张真实图 | FIG.3 | 缺少更多真实图的标注来做系统评估 |

**最关键的下一步**：标注 5-10 张真实专利图，用来微调 YOLO 和 CNN。根据经验，少量真实数据的微调通常能把召回率从 30% 级别提升到 70-80%。
