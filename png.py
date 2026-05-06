import os


class DatasetRenamer:
    def __init__(self):
        # ============ 在这里设置 ============
        self.base_dir = r"D:\codeArea\work\ros_ws(1)\train_main\high-voltage cable.yolov8\train"
        # self.base_dir = "/mnt/c/Users/OVERPASS/Desktop/model_wire/wire model nano.v11-2025-03-18-nano-dataset.yolov8/train"
        self.prefix = "G"  # 前缀，可以改成 B、C、Wire 等任何你想要的
        self.start_num = 1  # 起始编号
        self.num_digits = 4  # 编号位数，4位就是 A0001, A0002...
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


if __name__ == "__main__":
    renamer = DatasetRenamer()
    renamer.run()