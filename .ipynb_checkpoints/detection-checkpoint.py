import os


class DatasetChecker:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.images_dir = os.path.join(self.base_dir, "images")
        self.labels_dir = os.path.join(self.base_dir, "labels")

    def run(self):
        print(f"检查目录: {self.base_dir}")
        print("=" * 60)

        if not os.path.exists(self.labels_dir):
            print(f"错误：labels 目录不存在")
            return
        if not os.path.exists(self.images_dir):
            print(f"错误：images 目录不存在")
            return

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        images = set(
            os.path.splitext(f)[0] for f in os.listdir(self.images_dir)
            if f.lower().endswith(image_extensions)
        )
        labels = set(
            os.path.splitext(f)[0] for f in os.listdir(self.labels_dir)
            if f.endswith(".txt")
        )

        # 1. 图片和标注配对检查
        only_images = images - labels
        only_labels = labels - images
        paired = images & labels

        print(f"\n【1. 配对检查】")
        print(f"  图片总数: {len(images)}")
        print(f"  标注总数: {len(labels)}")
        print(f"  成功配对: {len(paired)}")
        if only_images:
            print(f"  ⚠️ 有图片无标注: {len(only_images)} 个")
            for name in sorted(list(only_images))[:5]:
                print(f"     {name}")
        if only_labels:
            print(f"  ⚠️ 有标注无图片: {len(only_labels)} 个")
            for name in sorted(list(only_labels))[:5]:
                print(f"     {name}")

        # 2. Class ID 统计
        class_count = {}
        total_boxes = 0
        bad_format = []
        segment_format = []
        out_of_range = []
        illegal_class = []
        empty_files = []

        txt_files = [f for f in os.listdir(self.labels_dir) if f.endswith(".txt")]

        for txt_name in txt_files:
            txt_path = os.path.join(self.labels_dir, txt_name)

            if os.path.getsize(txt_path) == 0:
                empty_files.append(txt_name)
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue

                    # 格式检查
                    if len(parts) < 5:
                        bad_format.append((txt_name, line_num, line.strip()))
                        continue
                    if len(parts) > 5:
                        segment_format.append((txt_name, line_num, len(parts)))
                        continue

                    # class_id 检查
                    class_id = parts[0]
                    if class_id not in ("0", "1", "2"):
                        illegal_class.append((txt_name, line_num, class_id))
                        continue

                    class_count[class_id] = class_count.get(class_id, 0) + 1
                    total_boxes += 1

                    # 坐标范围检查
                    try:
                        coords = [float(x) for x in parts[1:5]]
                        for val in coords:
                            if val < 0 or val > 1:
                                out_of_range.append((txt_name, line_num, line.strip()))
                                break
                    except ValueError:
                        bad_format.append((txt_name, line_num, line.strip()))

        # 输出结果
        print(f"\n【2. Class ID 分布】（共 {total_boxes} 个框）")
        names = {"0": "wire", "1": "pipe", "2": "rope"}
        for cid in ["0", "1", "2"]:
            count = class_count.get(cid, 0)
            status = "✅" if count > 0 else "❌ 缺失"
            print(f"  class {cid} ({names[cid]}): {count} 个框 {status}")

        print(f"\n【3. 格式检查】")
        if not bad_format:
            print(f"  ✅ 无格式错误")
        else:
            print(f"  ❌ {len(bad_format)} 行格式错误")
            for item in bad_format[:5]:
                print(f"     {item[0]} 第{item[1]}行: {item[2]}")

        print(f"\n【4. 分割格式残留（>5个值）】")
        if not segment_format:
            print(f"  ✅ 无分割格式残留")
        else:
            print(f"  ❌ {len(segment_format)} 行仍是分割格式")
            for item in segment_format[:5]:
                print(f"     {item[0]} 第{item[1]}行: {item[2]}个数值")

        print(f"\n【5. 非法 Class ID（不是0/1/2）】")
        if not illegal_class:
            print(f"  ✅ 无非法 class_id")
        else:
            print(f"  ❌ {len(illegal_class)} 行非法")
            for item in illegal_class[:5]:
                print(f"     {item[0]} 第{item[1]}行: class_id={item[2]}")

        print(f"\n【6. 坐标越界（超出0~1）】")
        if not out_of_range:
            print(f"  ✅ 无越界")
        else:
            print(f"  ❌ {len(out_of_range)} 行越界")
            for item in out_of_range[:5]:
                print(f"     {item[0]} 第{item[1]}行: {item[2]}")

        print(f"\n【7. 空文件】")
        if not empty_files:
            print(f"  ✅ 无空文件")
        else:
            print(f"  ⚠️ {len(empty_files)} 个空文件（负样本，通常没问题）")

        # 总结
        print(f"\n{'=' * 60}")
        all_good = (
            not bad_format and
            not segment_format and
            not illegal_class and
            not out_of_range and
            class_count.get("0", 0) > 0 and
            class_count.get("1", 0) > 0 and
            class_count.get("2", 0) > 0
        )
        if all_good:
            print("✅ 数据集检查通过，可以训练！")
        else:
            print("❌ 数据集有问题，请修复后再训练")


if __name__ == "__main__":
    DatasetChecker("/root/yolo_train/train_main/model_train/train").run()
    DatasetChecker("/root/yolo_train/train_main/model_train/valid").run()