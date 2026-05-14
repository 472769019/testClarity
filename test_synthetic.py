"""
测试脚本：构造不同质量的人脸图像，验证清晰度判断逻辑
"""
import cv2
import numpy as np
from face_quality import FaceQualityChecker


def make_synthetic_face(size=300, sharpness=1.0, brightness=128, contrast=1.0):
    """
    生成一张带"人脸"的合成图像（用 Haar 能检到的简易脸）
    sharpness: 1.0=原图, <1.0 = 模糊
    brightness: 平均亮度
    contrast: 对比度倍数
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 180  # 浅色背景

    # 画一张能被 Haar 检到的"脸"
    cx, cy = size // 2, size // 2
    # 脸部椭圆
    cv2.ellipse(img, (cx, cy), (90, 110), 0, 0, 360, (200, 180, 160), -1)
    # 眼睛（Haar 要求清晰的眼睛特征）
    cv2.circle(img, (cx - 35, cy - 20), 12, (255, 255, 255), -1)
    cv2.circle(img, (cx + 35, cy - 20), 12, (255, 255, 255), -1)
    cv2.circle(img, (cx - 35, cy - 20), 6, (30, 30, 30), -1)
    cv2.circle(img, (cx + 35, cy - 20), 6, (30, 30, 30), -1)
    # 眉毛
    cv2.line(img, (cx - 50, cy - 45), (cx - 20, cy - 40), (50, 30, 20), 4)
    cv2.line(img, (cx + 20, cy - 40), (cx + 50, cy - 45), (50, 30, 20), 4)
    # 鼻子
    cv2.line(img, (cx, cy - 10), (cx, cy + 20), (150, 120, 100), 3)
    # 嘴
    cv2.ellipse(img, (cx, cy + 50), (25, 8), 0, 0, 180, (100, 50, 50), 2)
    # 添加一点纹理（让清晰度差异更明显）
    noise = np.random.randint(0, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 应用对比度和亮度
    img = np.clip((img.astype(np.float32) - 128) * contrast + brightness, 0, 255).astype(np.uint8)

    # 应用模糊
    if sharpness < 1.0:
        k = int((1 - sharpness) * 25) | 1  # 奇数
        if k >= 3:
            img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def run_test_case(name, img, checker):
    report = checker.check(img)
    status = "✅ 可用" if report.usable else "❌ 不可用"
    print(f"\n[{name}] {status}  综合分={report.overall_score}")
    print(f"  原因: {report.reason}")
    print(f"  指标: lap={report.laplacian}  ten={report.tenengrad}  "
          f"亮度={report.brightness}  对比度={report.contrast}")
    print(f"  人脸: size={report.face_size}px  眼睛={report.eyes_detected}")
    return report


def main():
    # 关掉"必须检测到眼睛"，因为合成脸的眼睛形态和真实人脸有差异
    checker = FaceQualityChecker(require_eyes=False)

    print("=" * 60)
    print("人脸清晰度判断 - 合成数据测试")
    print("=" * 60)

    cases = [
        ("清晰正常", make_synthetic_face(sharpness=1.0, brightness=140, contrast=1.0)),
        ("轻微模糊", make_synthetic_face(sharpness=0.7, brightness=140, contrast=1.0)),
        ("严重模糊", make_synthetic_face(sharpness=0.2, brightness=140, contrast=1.0)),
        ("过暗", make_synthetic_face(sharpness=1.0, brightness=30, contrast=0.5)),
        ("过亮", make_synthetic_face(sharpness=1.0, brightness=240, contrast=0.4)),
        ("对比度低", make_synthetic_face(sharpness=1.0, brightness=140, contrast=0.2)),
    ]

    for name, img in cases:
        run_test_case(name, img, checker)

    # 测试无人脸场景
    print("\n" + "=" * 60)
    blank = np.ones((300, 300, 3), dtype=np.uint8) * 128
    run_test_case("无人脸（纯灰图）", blank, checker)


if __name__ == "__main__":
    main()
