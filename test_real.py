"""
端到端测试：在真实人脸照片上跑完整流程
从 GitHub OpenCV 仓库下载一张样例人脸图
"""
import os
import urllib.request
import cv2
from face_quality import FaceQualityChecker, draw_report


SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
SAMPLE_PATH = "/tmp/test_face.jpg"


def download_sample():
    if not os.path.exists(SAMPLE_PATH):
        print(f"下载样例图像...")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def main():
    path = download_sample()
    original = cv2.imread(path)
    print(f"原图尺寸: {original.shape}")

    checker = FaceQualityChecker(require_eyes=True)

    # 测试 1: 原图（清晰）
    print("\n" + "=" * 60)
    print("场景 1: 原始清晰图像")
    print("=" * 60)
    r = checker.check(original)
    print_report(r)
    cv2.imwrite("/home/claude/face_quality/out_sharp.jpg", draw_report(original, r))

    # 测试 2: 模糊
    print("\n" + "=" * 60)
    print("场景 2: 高斯模糊 (15x15)")
    print("=" * 60)
    blurred = cv2.GaussianBlur(original, (15, 15), 0)
    r = checker.check(blurred)
    print_report(r)
    cv2.imwrite("/home/claude/face_quality/out_blur.jpg", draw_report(blurred, r))

    # 测试 3: 过暗
    print("\n" + "=" * 60)
    print("场景 3: 过暗")
    print("=" * 60)
    dark = (original * 0.2).astype("uint8")
    r = checker.check(dark)
    print_report(r)

    # 测试 4: 过亮
    print("\n" + "=" * 60)
    print("场景 4: 过亮")
    print("=" * 60)
    import numpy as np
    bright = np.clip(original.astype("int16") + 100, 0, 255).astype("uint8")
    r = checker.check(bright)
    print_report(r)

    # 测试 5: 缩小（人脸变小）
    print("\n" + "=" * 60)
    print("场景 5: 缩小 4 倍（人脸变小）")
    print("=" * 60)
    small = cv2.resize(original, None, fx=0.25, fy=0.25)
    r = checker.check(small)
    print_report(r)

    # 测试 6: 运动模糊
    print("\n" + "=" * 60)
    print("场景 6: 运动模糊")
    print("=" * 60)
    import numpy as np
    kernel = np.zeros((15, 15))
    kernel[7, :] = 1.0 / 15
    motion = cv2.filter2D(original, -1, kernel)
    r = checker.check(motion)
    print_report(r)
    cv2.imwrite("/home/claude/face_quality/out_motion.jpg", draw_report(motion, r))


def print_report(r):
    status = "✅ 可用" if r.usable else "❌ 不可用"
    print(f"{status}  综合分: {r.overall_score}/100")
    print(f"原因: {r.reason}")
    if r.face_found:
        print(f"人脸框: {r.face_box}, 大小: {r.face_size}px, 眼睛: {r.eyes_detected}")
        print(f"清晰度: Laplacian={r.laplacian}, Tenengrad={r.tenengrad}")
        print(f"亮度: {r.brightness}, 对比度: {r.contrast}")
        print(f"过曝率: {r.overexposure * 100:.1f}%, 欠曝率: {r.underexposure * 100:.1f}%")


if __name__ == "__main__":
    main()
