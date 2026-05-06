import os


class LabelRemapper:
    def __init__(self):
        # ============ 设置区域 ============
        self.base_dir = "/mnt/c/Users/OVERPASS/Desktop/model_wire/Gas Tube- Flexible Hose.yolov8"

        # 目标类别：0=wire, 1=pipe, 2=rope
        # 原始数据集类别：
        # 0: Ball Valve
        # 1: Butane Can
        # 2: Butane Stove
        # 3: Cast Iron Gas Burners
        # 4: Flexible Hose       -> pipe
        # 5: Fusecock
        # 6: Gas Boiler
        # 7: Gas Meter
        # 8: Gas Tube            -> pipe
        # 9: Gas Vent Pipe       -> pipe
        # 10: Gas Water Heaters
        # 11: LPG Cylinders
        # 12: Pressure Regulator

        self.class_map = {
            4: 1,  # Flexible Hose -> pipe
            8: 1,  # Gas Tube -> pipe
            9: 1,  # Gas Vent Pipe -> pipe
        }

        # 不在 class_map 里的类别直接删除该行
        # 如果整张图删完了就变成空txt，保留作为负样本
        # ====================================

    def process_split(self, split_name):
        labels_dir = os.path.join(self.base_dir, split_name, "labels")

        if not os.path.exists(labels_dir):
            print(f"[跳过] {split_name}/labels 不存在")
            return

        txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        txt_files.sort()

        if not txt_files:
            print(f"[跳过] {split_name}/labels 下没有 txt 文件")
            return

        modified_count = 0
        empty_count = 0
        removed_lines_total = 0

        for txt_file in txt_files:
            txt_path = os.path.join(labels_dir, txt_file)

            with open(txt_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            removed_lines = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                original_class = int(parts[0])

                if original_class in self.class_map:
                    parts[0] = str(self.class_map[original_class])
                    new_lines.append(' '.join(parts))
                else:
                    removed_lines += 1

            # 写回文件（可能为空，作为负样本）
            with open(txt_path, 'w') as f:
                for new_line in new_lines:
                    f.write(new_line + '\n')

            removed_lines_total += removed_lines
            modified_count += 1

            if len(new_lines) == 0:
                empty_count += 1

        print(f"  [{split_name}] 处理 {modified_count} 个文件, "
              f"删除 {removed_lines_total} 行无关标注, "
              f"{empty_count} 个文件变为负样本(空标注)")

    def run(self):
        print(f"数据集路径: {self.base_dir}")
        print(f"类别映射: {self.class_map}")
        print(f"目标类别: 0=wire, 1=pipe, 2=rope")
        print("-" * 50)

        for split in ['train', 'valid', 'test']:
            self.process_split(split)

        print("-" * 50)
        print("完成。空标注文件已保留作为负样本。")


if __name__ == "__main__":
    remapper = LabelRemapper()
    remapper.run()