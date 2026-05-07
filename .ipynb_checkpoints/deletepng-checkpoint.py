import os
import random
import cv2


class VisualizeLabels:
    def __init__(self, base_dir, output_dir, sample_count=5):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.output_dir = output_dir
        self.sample_count = sample_count
        self.colors = {
            "0": (0, 255, 0),    # wire - 绿色
            "1": (0, 0, 255),    # pipe - 红色
            "2": (255, 0, 0),    # rope - 蓝色
        }
        self.names = {"0": "wire", "1": "pipe", "2": "rope"}

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        image_files = [
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(image_extensions)
        ]

        # 按前缀分组
        groups = {"B": [], "G": [], "R": []}
        for f in image_files:
            prefix = f[0].upper()
            if prefix in groups:
                groups[prefix].append(f)

        for prefix, files in groups.items():
            if not files:
                print(f"[{prefix}] 没有图片，跳过")
                continue

            samples = random.sample(files, min(self.sample_count, len(files)))
            for img_name in samples:
                self.draw_one(img_name, prefix)

        print(f"\n可视化结果保存在: {self.output_dir}")

    def draw_one(self, img_name, prefix):
        img_path = os.path.join(self.images_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        txt_path = os.path.join(self.labels_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取图片: {img_name}")
            return

        h, w = img.shape[:2]

        if not os.path.exists(txt_path):
            print(f"[跳过] 无标注: {img_name}")
            return

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = parts[0]
            x_center, y_center, bw, bh = [float(x) for x in parts[1:5]]

            # 转成像素坐标
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            color = self.colors.get(class_id, (255, 255, 255))
            label = self.names.get(class_id, f"class_{class_id}")

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = os.path.join(self.output_dir, f"{prefix}_{stem}.jpg")
        cv2.imwrite(out_path, img)
        print(f"[保存] {prefix}_{stem}.jpg ({len(lines)} 个框)")


if __name__ == "__main__":
    base_dir = r"/root/yolo_train/train_main/model_train/train"
    output_dir = r"/root/yolo_train/train_main/vis_check"

    vis = VisualizeLabels(base_dir, output_dir, sample_count=5)
    vis.run()