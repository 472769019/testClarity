"""
人脸图像清晰度判断
================
多指标融合：清晰度 + 自然度(块状伪影) + 亮度 + 对比度 + 人脸大小 + 姿态

依赖:
    pip install opencv-python opencv-contrib-python numpy
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List


# ============================================================
# 1. 清晰度指标（针对 ROI 计算）
# ============================================================

def laplacian_score(gray: np.ndarray) -> float:
    """Laplacian 方差：越大越清晰。最常用、最快"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad_score(gray: np.ndarray) -> float:
    """Tenengrad：基于 Sobel 梯度，对噪声更鲁棒"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def brenner_score(gray: np.ndarray) -> float:
    """Brenner 梯度：相邻像素差平方和，速度最快"""
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float(np.mean(diff * diff))


def fft_high_freq_ratio(gray: np.ndarray, cutoff_ratio: float = 0.15) -> float:
    """频域法：高频能量占比。清晰图像高频更多"""
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = h // 2, w // 2
    ry, rx = int(h * cutoff_ratio), int(w * cutoff_ratio)
    mask = np.ones_like(mag, dtype=bool)
    mask[cy - ry:cy + ry, cx - rx:cx + rx] = False

    total = mag.sum() + 1e-8
    high = mag[mask].sum()
    return float(high / total)


# ============================================================
# 2. 辅助指标
# ============================================================

def brightness_score(gray: np.ndarray) -> float:
    """平均亮度 [0, 255]，太暗或太亮都不行"""
    return float(gray.mean())


def contrast_score(gray: np.ndarray) -> float:
    """对比度（标准差）"""
    return float(gray.std())


def overexposure_ratio(gray: np.ndarray, thr: int = 240) -> float:
    """过曝像素占比"""
    return float((gray > thr).mean())


def underexposure_ratio(gray: np.ndarray, thr: int = 15) -> float:
    """欠曝像素占比"""
    return float((gray < thr).mean())


def block_artifact_ratio(gray: np.ndarray) -> float:
    """JPEG块状伪影检测：测量8px对齐网格边界处的梯度不连续性。

    原理：JPEG压缩以8×8像素为DCT块单位，压缩伪影会在块边界产生系统性梯度增强。
    服装条纹、发型、室外背景等自然纹理不具有这种对8px网格对齐的系统性特征。
    通过枚举全部8种偏移（0-7px）找出使边界/内部梯度比最大的对齐方式，
    取边界梯度均值与内部梯度均值之比的超出量作为伪影强度。

    返回值 [0, 1]：0=无伪影，>0.3=明显块伪影，>0.5=严重块状/像素化。
    自然图像通常 <0.2；重度JPEG压缩或低分辨率上采样通常 >0.35。
    """
    img = gray.astype(np.float64)
    h, w = img.shape
    if h < 32 or w < 32:
        return 0.0

    h_grad = np.abs(np.diff(img, axis=1))  # shape (h, w-1)
    v_grad = np.abs(np.diff(img, axis=0))  # shape (h-1, w)

    best_ratio = 1.0  # 基准：无伪影时比值≈1.0

    for offset in range(8):
        # 8px JPEG块边界位置（遍历所有8种对齐偏移）
        bnd_cols = list(range(offset + 7, w - 1, 8))
        bnd_rows = list(range(offset + 7, h - 1, 8))

        if len(bnd_cols) >= 2:
            int_cols = [i for i in range(w - 1) if (i - offset - 7) % 8 != 0]
            if int_cols:
                bm = h_grad[:, bnd_cols].mean()
                im = h_grad[:, int_cols].mean()
                if im > 1e-6:
                    best_ratio = max(best_ratio, bm / im)

        if len(bnd_rows) >= 2:
            int_rows = [i for i in range(h - 1) if (i - offset - 7) % 8 != 0]
            if int_rows:
                bm = v_grad[bnd_rows, :].mean()
                im = v_grad[int_rows, :].mean()
                if im > 1e-6:
                    best_ratio = max(best_ratio, bm / im)

    # best_ratio=1.0 → 无伪影(返回0)；越高伪影越重
    return float(min(max(0.0, best_ratio - 1.0), 1.0))


# ============================================================
# 3. 人脸检测
# ============================================================

class FaceDetector:
    """
    多策略 Haar 级联检测，逐步放宽参数直到找到人脸。
    支持对过曝/低对比图像做 CLAHE 增强后再检测。
    """

    def __init__(self):
        base = cv2.data.haarcascades
        self.detector = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        # alt2 对某些角度和光照更鲁棒
        self.detector_alt2 = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self.eye_detector = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _try_detect(self, gray: np.ndarray, detector) -> List:
        """在给定图像上依次尝试多组参数，返回首个非空结果"""
        for scale, neighbors, min_sz in [
            (1.10, 5, (50, 50)),
            (1.05, 4, (40, 40)),
            (1.05, 3, (30, 30)),
            (1.03, 2, (25, 25)),
            (1.02, 1, (20, 20)),
        ]:
            faces = detector.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_sz
            )
            if len(faces) == 0:
                continue
            # 过滤掉面积不足图像 0.5% 的小噪声框
            area_thr = gray.shape[0] * gray.shape[1] * 0.005
            valid = [f for f in faces if f[2] * f[3] >= area_thr]
            if valid:
                return valid
        return []

    def detect(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """多策略检测，返回 [(x, y, w, h), ...]，按面积从大到小排序"""
        # 策略1：default cascade，原图
        faces = self._try_detect(gray, self.detector)
        if not faces:
            # 策略2：alt2 cascade，原图
            faces = self._try_detect(gray, self.detector_alt2)
        if not faces:
            # 策略3：CLAHE 增强后再检测（对过曝/低对比图像有效）
            enhanced = self._clahe.apply(gray)
            faces = self._try_detect(enhanced, self.detector)
        if not faces:
            faces = self._try_detect(enhanced, self.detector_alt2)

        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return [tuple(map(int, f)) for f in faces]

    def detect_eyes(self, face_gray: np.ndarray) -> int:
        """在人脸区域内检测眼睛数量，用于粗略判断姿态/遮挡"""
        eyes = self.eye_detector.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
        )
        return len(eyes)


# ============================================================
# 4. 主类：融合判断
# ============================================================

@dataclass
class QualityReport:
    """质量评估结果"""
    usable: bool                  # 最终判定：能不能用
    reason: str                   # 判定原因（不能用时给出具体问题）
    face_found: bool              # 是否检测到人脸
    face_box: Optional[Tuple[int, int, int, int]]

    # 各项指标
    laplacian: float
    tenengrad: float
    brenner: float
    fft_ratio: float
    brightness: float
    contrast: float
    overexposure: float
    underexposure: float
    face_size: int                # 人脸短边像素
    eyes_detected: int
    block_artifact: float         # 块状伪影自相关系数，0=无伪影，越大越严重

    # 综合分数 [0, 100]，越高越好
    overall_score: float

    def to_dict(self):
        return asdict(self)


class FaceQualityChecker:
    """人脸图像质量综合判断器"""

    def __init__(
        self,
        # 清晰度阈值（在人脸 ROI 上）
        # laplacian_thr=15：经验值，光滑皮肤年轻人/AI人脸的清晰证件照 lap 约 18-80，
        # 真正模糊的图像通常 lap < 12
        laplacian_thr: float = 15.0,
        tenengrad_thr: float = 300.0,
        # 亮度阈值
        bright_min: float = 50.0,
        bright_max: float = 220.0,
        # 对比度
        contrast_min: float = 20.0,
        # 过/欠曝
        over_max: float = 0.20,
        under_max: float = 0.30,
        # 人脸大小
        min_face_size: int = 80,
        # 是否要求检测到眼睛
        require_eyes: bool = False,
        # 块状伪影阈值（8px边界/内部梯度比的超出量）
        # 自然图像通常 <0.20，明显JPEG/像素化伪影通常 >0.35
        block_artifact_thr: float = 0.35,
    ):
        self.laplacian_thr = laplacian_thr
        self.tenengrad_thr = tenengrad_thr
        self.bright_min = bright_min
        self.bright_max = bright_max
        self.contrast_min = contrast_min
        self.over_max = over_max
        self.under_max = under_max
        self.min_face_size = min_face_size
        self.require_eyes = require_eyes
        self.block_artifact_thr = block_artifact_thr

        self.detector = FaceDetector()

    def _compute_overall(self, lap, ten, bright, contrast, over, under, face_size, block_artifact) -> float:
        """加权综合分：清晰度40% + 自然度20% + 亮度10% + 对比度10% + 曝光10% + 人脸大小10%"""
        # 清晰度（40%）
        # 饱和点设为 laplacian_thr * 4，保证清晰图像接近满分
        s_lap = min(lap / (self.laplacian_thr * 4), 1.0)
        s_ten = min(ten / (self.tenengrad_thr * 4), 1.0)
        s_sharp = (s_lap + s_ten) / 2

        # 自然度（20%）：JPEG块状伪影越少越好
        # 新指标从0开始（0=无伪影），超过_good开始线性扣分，达到_bad时归零
        _good = 0.10   # 低于此值视为正常，不扣分
        _bad = self.block_artifact_thr
        s_natural = max(0.0, 1.0 - (block_artifact - _good) / max(_bad - _good, 0.01))
        s_natural = min(1.0, s_natural)

        # 亮度（10%）：离中心越近越好
        s_bright = 1.0 - abs(bright - 135) / 135
        s_bright = max(0.0, min(1.0, s_bright))

        # 对比度（10%）
        s_contrast = min(contrast / 60, 1.0)

        # 曝光（10%）
        s_expo = 1.0 - min(over + under * 0.5, 1.0)

        # 人脸大小（10%）
        s_size = min(face_size / (self.min_face_size * 3), 1.0)

        score = (
            s_sharp * 40
            + s_natural * 20
            + s_bright * 10
            + s_contrast * 10
            + s_expo * 10
            + s_size * 10
        )
        return round(score, 2)

    def check(self, image: np.ndarray) -> QualityReport:
        """
        输入：BGR 图像（cv2.imread 的结果）
        输出：QualityReport
        """
        if image is None or image.size == 0:
            raise ValueError("空图像")

        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 人脸检测
        faces = self.detector.detect(gray_full)
        if not faces:
            return QualityReport(
                usable=False, reason="未检测到人脸",
                face_found=False, face_box=None,
                laplacian=0, tenengrad=0, brenner=0, fft_ratio=0,
                brightness=float(gray_full.mean()), contrast=float(gray_full.std()),
                overexposure=0, underexposure=0,
                face_size=0, eyes_detected=0, block_artifact=0.0, overall_score=0,
            )

        # 取最大人脸
        x, y, w, h = faces[0]
        face_gray = gray_full[y:y + h, x:x + w]
        face_size = min(w, h)

        # 2. 在人脸 ROI 上计算各项指标
        lap = laplacian_score(face_gray)
        ten = tenengrad_score(face_gray)
        bre = brenner_score(face_gray)
        fft = fft_high_freq_ratio(face_gray)
        bright = brightness_score(face_gray)
        contrast = contrast_score(face_gray)
        over = overexposure_ratio(face_gray)
        under = underexposure_ratio(face_gray)
        eyes = self.detector.detect_eyes(face_gray)
        block = block_artifact_ratio(face_gray)

        # 3. 综合判定
        reasons = []
        if face_size < self.min_face_size:
            reasons.append(f"人脸太小({face_size}px<{self.min_face_size})")
        if lap < self.laplacian_thr:
            reasons.append(f"图像模糊(lap={lap:.1f}<{self.laplacian_thr})")
        if bright < self.bright_min:
            reasons.append(f"过暗(亮度={bright:.1f})")
        elif bright > self.bright_max:
            reasons.append(f"过亮(亮度={bright:.1f})")
        if contrast < self.contrast_min:
            reasons.append(f"对比度低({contrast:.1f})")
        if over > self.over_max:
            reasons.append(f"过曝({over * 100:.1f}%)")
        if under > self.under_max:
            reasons.append(f"欠曝({under * 100:.1f}%)")
        if self.require_eyes and eyes < 1:
            reasons.append("未检测到眼睛(可能侧脸/遮挡)")
        if block > self.block_artifact_thr:
            reasons.append(f"像素化/块状伪影明显(acf={block:.3f})")

        usable = len(reasons) == 0
        reason = "通过" if usable else "; ".join(reasons)

        overall = self._compute_overall(lap, ten, bright, contrast, over, under, face_size, block)

        return QualityReport(
            usable=usable, reason=reason,
            face_found=True, face_box=(x, y, w, h),
            laplacian=round(lap, 2), tenengrad=round(ten, 2),
            brenner=round(bre, 2), fft_ratio=round(fft, 4),
            brightness=round(bright, 2), contrast=round(contrast, 2),
            overexposure=round(over, 4), underexposure=round(under, 4),
            face_size=face_size, eyes_detected=eyes,
            block_artifact=round(block, 4),
            overall_score=overall,
        )

    def check_file(self, path: str) -> QualityReport:
        """便捷接口：直接传文件路径"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return self.check(img)


# ============================================================
# 5. 命令行入口 + 可视化
# ============================================================

def draw_report(image: np.ndarray, report: QualityReport) -> np.ndarray:
    """在图上画出框和评分"""
    out = image.copy()
    if report.face_box:
        x, y, w, h = report.face_box
        color = (0, 200, 0) if report.usable else (0, 0, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    label = f"{'OK' if report.usable else 'NG'}  score={report.overall_score}"
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 200, 0) if report.usable else (0, 0, 220), 2)
    cv2.putText(out, f"lap={report.laplacian} ten={report.tenengrad} blk={report.block_artifact}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if not report.usable:
        cv2.putText(out, report.reason[:60], (10, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    return out


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("用法: python face_quality.py <image_path> [output_vis_path]")
        sys.exit(1)

    checker = FaceQualityChecker()
    report = checker.check_file(sys.argv[1])
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    if len(sys.argv) >= 3:
        img = cv2.imread(sys.argv[1])
        vis = draw_report(img, report)
        cv2.imwrite(sys.argv[2], vis)
        print(f"可视化已保存到: {sys.argv[2]}")
