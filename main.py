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
    print(f"[{path}] {status}，综合分 {report.overall_score}")
    print(f"  清晰度  lap={report.laplacian:.1f}  ten={report.tenengrad:.1f}")
    print(f"  块状伪影 blk={report.block_artifact:.3f}  (>={checker.block_artifact_thr} 不可用)")
    print(f"  亮度={report.brightness:.1f}  对比度={report.contrast:.1f}")
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
print("=" * 50)
print(f"总计: {total}  通过: {pass_count}  不通过: {fail_count}  通过率: {pass_count/total*100:.1f}%")
if fail_reasons:
    print("不通过原因分布:")
    for reason, count in fail_reasons.most_common():
        print(f"  {reason}: {count}")
