# 人脸图像质量检测工具

面向**证件照**场景的人脸图像质量综合判断器。纯传统图像算法实现，无需 GPU、无需训练数据、毫秒级响应。

## 核心能力

| 检测项 | 说明 |
|--------|------|
| 模糊检测 | Laplacian 方差，对光滑皮肤/AI 人脸做了自适应校准 |
| 块状伪影 | JPEG 8px 网格边界梯度比，区分真实压缩伪影与自然纹理 |
| 亮度/曝光 | 过曝、欠曝像素占比 |
| 对比度 | 人脸 ROI 标准差 |
| 人脸检测 | 多级 Haar 级联回退 + CLAHE 增强，支持光滑/AI/儿童人脸 |

## 文件说明

```
face_quality.py     主模块，包含所有算法和 FaceQualityChecker 类
main.py             批量跑 ./data 目录并输出统计
test_metrics.py     验证清晰度指标对模糊的响应
test_real.py        端到端测试样例（需要真实人脸图）
test_synthetic.py   合成模糊/伪影测试
```

## 安装

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install opencv-contrib-python numpy
```

> 注意：需要 `opencv-contrib-python`，不能只装 `opencv-python`（需要其中包含的额外级联分类器）。

## 快速使用

```python
from face_quality import FaceQualityChecker

checker = FaceQualityChecker()
report = checker.check_file("photo.jpg")

if report.usable:
    print(f"可用，综合分 {report.overall_score}")
else:
    print(f"不可用: {report.reason}")

# 获取所有字段
print(report.to_dict())
```

## 输出字段（QualityReport）

| 字段 | 类型 | 说明 |
|------|------|------|
| `usable` | bool | 最终判定：True / False |
| `reason` | str | 失败原因（多个用 `; ` 分隔），通过时为 `"通过"` |
| `overall_score` | float | 综合分 0~100 |
| `face_found` | bool | 是否检测到人脸 |
| `face_box` | tuple | 人脸框 `(x, y, w, h)`，未检测到时为 `None` |
| `laplacian` | float | Laplacian 方差（清晰度主指标，越大越清晰） |
| `tenengrad` | float | Sobel 梯度能量（辅助清晰度，对噪声鲁棒） |
| `brenner` | float | Brenner 梯度（最快的清晰度近似） |
| `fft_ratio` | float | 高频能量占比（频域清晰度） |
| `brightness` | float | 人脸区域平均亮度 0~255 |
| `contrast` | float | 人脸区域对比度（标准差） |
| `overexposure` | float | 过曝像素占比 0~1 |
| `underexposure` | float | 欠曝像素占比 0~1 |
| `face_size` | int | 人脸短边像素数 |
| `eyes_detected` | int | 检测到的眼睛数（粗略反映正脸/姿态） |
| `block_artifact` | float | JPEG 块状伪影强度 0~1（0=无伪影，>0.35=明显） |

## 综合评分构成

```
综合分 = 清晰度 40% + 自然度 20% + 亮度 10% + 对比度 10% + 曝光 10% + 人脸大小 10%
```

- **清晰度**：Laplacian 和 Tenengrad 均值（各 20%）
- **自然度**：JPEG 块状伪影越少越高

## 阈值说明与调优

默认阈值针对**证件照**场景经过实测标定，一般不需要修改。如有必要：

```python
checker = FaceQualityChecker(
    laplacian_thr=15,         # 模糊阈值。光滑皮肤清晰证件照 lap≈18-80，真正模糊 lap<12
    tenengrad_thr=300,        # Tenengrad 参考值（仅用于评分，不作为硬判据）
    bright_min=50,            # 人脸最低平均亮度
    bright_max=220,           # 人脸最高平均亮度
    contrast_min=20,          # 最低对比度（标准差）
    over_max=0.20,            # 最大过曝像素占比
    under_max=0.30,           # 最大欠曝像素占比
    min_face_size=80,         # 人脸短边最小像素（活体检测建议 ≥120）
    require_eyes=False,       # True=必须检测到眼睛，会误杀佩戴眼镜/半闭眼的图片
    block_artifact_thr=0.35,  # 块状伪影阈值。自然图像 <0.20，重度 JPEG/像素化 >0.35
)
```

**标定流程**：

1. 准备 50~100 张明确"可用"和"不可用"的样本
2. 跑 `checker.check()` 获取各项指标值
3. 按业务需求找分布边界，调整对应参数

## 命令行用法

```bash
# 输出 JSON 报告
python face_quality.py photo.jpg

# 同时生成可视化结果（人脸框 + 评分叠加）
python face_quality.py photo.jpg output_vis.jpg
```

```bash
# 批量处理 ./data 目录，输出统计
python main.py
```

## 算法说明

### 模糊检测

使用 **Laplacian 方差**作为主指标。与通常使用的阈值 100 不同，本项目将阈值设为 15，原因：

- 有皮肤纹理的成熟人脸：`lap ≈ 80~500`
- 光滑皮肤的年轻人 / AI 人脸（清晰状态）：`lap ≈ 18~80`
- 真正模糊的图像：`lap < 12`

阈值 100 会将大量清晰的光滑人脸误判为模糊，15 是更通用的下限。

### 块状伪影检测

JPEG 压缩以 8×8 像素为 DCT 块单位，重度压缩时块边界会出现明显梯度不连续。

**方法**：枚举全部 8 种像素偏移（0~7px），对每种偏移测量 8px 对齐边界处与内部的梯度均值之比，取最大值减 1 作为伪影强度。

- 自然图像（服装条纹、头发纹理、室外背景）：`blk ≈ 0.01~0.15`
- 重度 JPEG 压缩 / 低分辨率上采样：`blk > 0.35`

这与早期的自相关方法（ACF）相比，不会被服装条纹、帽子图案、头发等自然周期纹理误触发。

### 人脸检测

多级回退策略：

1. `haarcascade_frontalface_default`，标准参数
2. 同上，逐步放宽 `scaleFactor` / `minNeighbors`（共 5 级）
3. `haarcascade_frontalface_alt2`，对部分角度更鲁棒
4. CLAHE 直方图增强后重试（改善过曝/低对比度图像的检测率）

## 性能参考

| 图像尺寸 | 典型耗时（CPU） |
|----------|----------------|
| 640×480  | 5~20ms |
| 1280×960 | 15~50ms |

Haar 人脸检测占约 60% 耗时。如需更高吞吐，可换用 OpenCV 内置的 YuNet DNN 检测器（需下载 `.onnx` 模型）。

## 已知局限

- Haar 级联对**强侧脸**（偏转 >45°）检测率低，建议换用 YuNet / SCRFD
- 重度美颜滤镜会将 Laplacian 压低至接近阈值，可能误判
- 运动模糊与失焦模糊不区分（业务上通常不需要）
- 极强逆光需要额外的曝光补偿预处理

## 进阶替换方向

1. **人脸检测**：`cv2.FaceDetectorYN`（YuNet，OpenCV 4.8+）精度显著更高
2. **姿态估计**：增加关键点检测，量化 pitch / yaw / roll
3. **遮挡检测**：基于关键点置信度判断眼睛/嘴巴是否被遮挡
4. **深度质量模型**：叠加 FaceQnet / SER-FIQ 做最终质量打分
