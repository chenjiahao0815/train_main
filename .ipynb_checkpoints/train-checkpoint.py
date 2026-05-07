import os
import random
import shutil


class SplitDataset:
    def __init__(self):
        # ============ 设置区域 ============
        self.base_dir = r"D:\codeArea\work\ros_ws(1)\train_main\Rope Detection.yolov8"
        self.val_ratio = 0.2  # 20% 作为验证集
        self.random_seed = 42  # 固定随机种子，保证可复现
        # ====================================

        self.train_images = os.path.join(self.base_dir, "train", "images")
        self.train_labels = os.path.join(self.base_dir, "train", "labels")
        self.val_images = os.path.join(self.base_dir, "valid", "images")
        self.val_labels = os.path.join(self.base_dir, "valid", "labels")

    def run(self):
        # 创建 valid 目录
        os.makedirs(self.val_images, exist_ok=True)
        os.makedirs(self.val_labels, exist_ok=True)

        # 获取所有图片
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        image_files = [
            f for f in os.listdir(self.train_images)
            if f.lower().endswith(image_extensions)
        ]
        image_files.sort()

        # 随机抽取
        random.seed(self.random_seed)
        val_count = int(len(image_files) * self.val_ratio)
        val_files = random.sample(image_files, val_count)

        print(f"总图片数: {len(image_files)}")
        print(f"移动到 valid: {val_count}")
        print(f"保留在 train: {len(image_files) - val_count}")
        print("-" * 50)

        moved = 0
        for img_file in val_files:
            stem = os.path.splitext(img_file)[0]

            # 移动图片
            src_img = os.path.join(self.train_images, img_file)
            dst_img = os.path.join(self.val_images, img_file)
            shutil.move(src_img, dst_img)

            # 移动对应标注
            txt_file = stem + ".txt"
            src_txt = os.path.join(self.train_labels, txt_file)
            dst_txt = os.path.join(self.val_labels, txt_file)
            if os.path.exists(src_txt):
                shutil.move(src_txt, dst_txt)

            moved += 1

        print(f"完成，移动了 {moved} 组文件到 valid/")


if __name__ == "__main__":
    splitter = SplitDataset()
    splitter.run()