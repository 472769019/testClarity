from face_quality import FaceQualityChecker
import os
import collections

checker = FaceQualityChecker()

directory = "./data"
fail_reasons = collections.Counter()
total = pass_count = 0

for name in sorted(os.listdir(directory)):
    path = directory + "/" + name
    try:
        report = checker.check_file(path)
    except Exception as e:
        print(f"[{path}] 读取失败: {e}")
        continue

    total += 1
    status = "可用" if report.usable else "不可用"
    print(f"[{path}] {status}，综合分 {report.overall_score:.2f}")
    print(
        f"  清晰度 {report.score_sharpness:5.2f}/40  "
        f"自然度 {report.score_naturalness:5.2f}/20  "
        f"亮度 {report.score_brightness:5.2f}/10  "
        f"对比度 {report.score_contrast:5.2f}/10  "
        f"曝光 {report.score_exposure:5.2f}/10  "
        f"人脸 {report.score_face_size:5.2f}/10"
    )
    print(
        f"  lap={report.laplacian:.1f}  edge={report.edge_sharpness:.1f}  "
        f"ten={report.tenengrad:.1f}  "
        f"blk={report.block_artifact:.3f}(>={checker.block_artifact_thr})  "
        f"亮度={report.brightness:.1f}  对比度={report.contrast:.1f}"
    )
    if not report.usable:
        print(f"  原因: {report.reason}")
    print()

    if report.usable:
        pass_count += 1
    else:
        for part in report.reason.split("; "):
            key = part.split("(")[0].strip()
            fail_reasons[key] += 1

fail_count = total - pass_count
print("=" * 60)
print(f"总计: {total}  通过: {pass_count}  不通过: {fail_count}  通过率: {pass_count / total * 100:.1f}%")
if fail_reasons:
    print("不通过原因分布:")
    for reason, count in fail_reasons.most_common():
        print(f"  {reason}: {count}")
