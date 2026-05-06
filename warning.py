import os


class DatasetFixer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.labels_dir = os.path.join(self.base_dir, "labels")

    def fix_segment_to_bbox(self):
        """把多边形分割格式转成 bbox 格式"""
        if not os.path.exists(self.labels_dir):
            print(f"错误：labels 目录不存在: {self.labels_dir}")
            return

        txt_files = [f for f in os.listdir(self.labels_dir) if f.endswith(".txt")]
        fixed_count = 0

        for txt_name in txt_files:
            txt_path = os.path.join(self.labels_dir, txt_name)
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            file_changed = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue

                if len(parts) > 5:
                    # 这是分割格式，转成 bbox
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]

                    # 提取所有 x 和 y 坐标（交替排列：x1 y1 x2 y2 ...）
                    xs = coords[0::2]
                    ys = coords[1::2]

                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    # 计算 bbox
                    x_min = min(xs)
                    x_max = max(xs)
                    y_min = min(ys)
                    y_max = max(ys)

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    file_changed = True
                else:
                    # 已经是 bbox 格式，保持不变
                    new_lines.append(line)

            if file_changed:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                fixed_count += 1
                print(f"[转换] {txt_name}")

        print(f"\n分割→bbox 转换完成：{fixed_count} 个文件")

    def check_class_distribution(self):
        """统计各 class_id 的数量"""
        txt_files = [f for f in os.listdir(self.labels_dir) if f.endswith(".txt")]
        class_count = {}

        for txt_name in txt_files:
            txt_path = os.path.join(self.labels_dir, txt_name)
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = parts[0]
                        class_count[cid] = class_count.get(cid, 0) + 1

        print("\nClass ID 分布：")
        for cid in sorted(class_count.keys()):
            print(f"  class {cid}: {class_count[cid]} 个框")


if __name__ == "__main__":
    # ============ 处理 train ============
    print("=" * 50)
    print("处理 train 目录")
    print("=" * 50)
    train_fixer = DatasetFixer(r"D:\codeArea\work\ros_ws(1)\train_main\model_train\train")
    train_fixer.fix_segment_to_bbox()
    train_fixer.check_class_distribution()

    # ============ 处理 valid ============
    print("\n" + "=" * 50)
    print("处理 valid 目录")
    print("=" * 50)
    valid_fixer = DatasetFixer(r"D:\codeArea\work\ros_ws(1)\train_main\model_train\valid")
    valid_fixer.fix_segment_to_bbox()
    valid_fixer.check_class_distribution()