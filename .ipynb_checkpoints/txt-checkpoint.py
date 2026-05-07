import os


class DatasetRenamer:
    def __init__(self, base_dir):
        # ============ 在这里设置 ============
        self.base_dir = r'D:\codeArea\work\ros_ws(1)\train_main\Rope Detection.yolov8\train'
        self.prefix = "R"  # R 代表 rope（绳子）
        self.start_num = 1  # 起始编号
        self.num_digits = 4  # 编号位数，R0001, R0002...
        # ====================================

        self.images_dir = os.path.join(self.base_dir, "images")
        self.labels_dir = os.path.join(self.base_dir, "labels")

    def run(self):
        # 检查目录是否存在
        if not os.path.exists(self.images_dir):
            print(f"错误：images 目录不存在: {self.images_dir}")
            return
        if not os.path.exists(self.labels_dir):
            print(f"错误：labels 目录不存在: {self.labels_dir}")
            return

        # 获取所有图片文件，按文件名排序
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        image_files = [
            f for f in os.listdir(self.images_dir)
            if os.path.isfile(os.path.join(self.images_dir, f)) and f.lower().endswith(image_extensions)
        ]
        image_files.sort()

        if not image_files:
            print("images 目录下没有找到图片文件")
            return

        print(f"共找到 {len(image_files)} 张图片")
        print(f"命名规则: {self.prefix}{str(self.start_num).zfill(self.num_digits)} 开始")
        print("-" * 50)

        success_count = 0
        skip_count = 0

        for i, image_filename in enumerate(image_files):
            # 生成新名字
            num = self.start_num + i
            new_name = f"{self.prefix}{str(num).zfill(self.num_digits)}"

            # 图片原始信息
            image_stem, image_ext = os.path.splitext(image_filename)
            old_image_path = os.path.join(self.images_dir, image_filename)
            new_image_path = os.path.join(self.images_dir, new_name + image_ext)

            # 根据图片原始名字去找对应的 txt
            old_label_path = os.path.join(self.labels_dir, image_stem + ".txt")
            new_label_path = os.path.join(self.labels_dir, new_name + ".txt")

            # 检查对应的 txt 是否存在
            if not os.path.exists(old_label_path):
                print(f"[跳过] 图片 {image_filename} 没有找到对应的标注文件 {image_stem}.txt")
                skip_count += 1
                continue

            # 先重命名图片，再重命名对应的 txt
            os.rename(old_image_path, new_image_path)
            os.rename(old_label_path, new_label_path)

            print(f"[完成] {image_filename} -> {new_name}{image_ext}  |  {image_stem}.txt -> {new_name}.txt")
            success_count += 1

        print("-" * 50)
        print(f"完成: {success_count} 组, 跳过: {skip_count} 组")

    def fix_class_id(self, old_id="0", new_id="2"):
        """把标注文件里的 class_id 从 old_id 改成 new_id"""
        if not os.path.exists(self.labels_dir):
            print(f"错误：labels 目录不存在: {self.labels_dir}")
            return

        txt_files = [f for f in os.listdir(self.labels_dir) if f.endswith(".txt")]
        changed_count = 0

        for txt_name in txt_files:
            txt_path = os.path.join(self.labels_dir, txt_name)
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            file_changed = False
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == old_id:
                    parts[0] = new_id
                    file_changed = True
                new_lines.append(" ".join(parts) + "\n")

            if file_changed:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                changed_count += 1

        print(f"class_id 修改完成：{changed_count} 个文件中 {old_id} -> {new_id}")


if __name__ == "__main__":
    # ============ 修改这里的路径，指向 rope 数据集的 train 目录 ============
    base_dir = r"D:\codeArea\work\ros_ws(1)\train_main\rope.yolov8\train"
    # ====================================================================

    renamer = DatasetRenamer(base_dir)

    # 第一步：重命名文件（R0001, R0002...）
    renamer.run()

    # 第二步：把 class_id 从 0（原数据集只有一类 Rope）改成 2（你最终的 rope 位置）
    renamer.fix_class_id(old_id="0", new_id="2")