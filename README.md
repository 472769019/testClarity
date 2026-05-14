# 人脸图像清晰度判断工具

纯传统图像算法实现，无需 GPU、无需训练数据、毫秒级响应。

## 文件说明

- `face_quality.py` —— 主模块，包含所有算法和 `FaceQualityChecker` 类
- `test_metrics.py` —— 验证清晰度指标对模糊的响应
- `test_real.py` —— 端到端测试样例（需要真实人脸图）

## 安装

```bash
pip install opencv-python numpy
```

## 快速使用

```python
from face_quality import FaceQualityChecker

checker = FaceQualityChecker()
report = checker.check_file("photo.jpg")

if report.usable:
    print(f"可用，综合分 {report.overall_score}")
else:
    print(f"不可用: {report.reason}")
```

## 输出字段

`QualityReport` 对象包含：

| 字段 | 说明 |
|---|---|
| `usable` | 最终判定：True / False |
| `reason` | 失败原因（多个用 `;` 分隔）|
| `overall_score` | 综合分 0~100 |
| `face_box` | 人脸框 (x, y, w, h) |
| `laplacian` | Laplacian 方差（清晰度，越大越清晰）|
| `tenengrad` | Sobel 梯度能量（清晰度，对噪声鲁棒）|
| `brenner` | Brenner 梯度（最快）|
| `fft_ratio` | 高频能量占比 |
| `brightness` | 平均亮度 0~255 |
| `contrast` | 对比度（标准差）|
| `overexposure` / `underexposure` | 过/欠曝像素占比 |
| `face_size` | 人脸短边像素 |
| `eyes_detected` | 检测到的眼睛数（粗略反映姿态）|

## 阈值调优

默认阈值是通用值，**必须用你自己的数据标定**。流程：

1. 准备 50~100 张明确"可用"和"不可用"的样本
2. 跑 `checker.check()` 拿到各项指标
3. 看分布，定阈值。通常重点调这几个：

```python
checker = FaceQualityChecker(
    laplacian_thr=100,    # 模糊阈值，标准照片用 100~150，监控/手机自拍可能要 50~80
    tenengrad_thr=500,    # 配合 lap 用
    min_face_size=80,     # 人脸识别一般要求 ≥80px，活体检测 ≥120px
    bright_min=50,
    bright_max=220,
    require_eyes=True,    # 严格场景开启，宽松场景关闭
)
```

## 命令行用法

```bash
# 输出 JSON 报告
python face_quality.py photo.jpg

# 同时生成可视化结果（带框 + 评分）
python face_quality.py photo.jpg output_vis.jpg
```

## 工程建议

**性能**：单张 640x480 图像在 CPU 上耗时约 5~15ms（Haar 检测占大头）。
如需更高准确率，把 `FaceDetector` 换成 RetinaFace / SCRFD。

**改进方向**：
1. 把 Haar 换成 DNN 人脸检测（OpenCV 自带 `face_detection_yunet`）
2. 增加关键点检测，做姿态估计（pitch/yaw/roll）
3. 增加遮挡检测（眼睛/嘴巴关键点置信度）
4. 业务关键场景叠加深度学习质量模型（FaceQnet / SER-FIQ）

**已知局限**：
- 运动模糊和失焦模糊不区分（业务上一般也不需要）
- 极小幅度的模糊难以稳定区分
- 强逆光场景需要单独处理
