import os

# ============ 修改这里定位到 train 目录 ============
TRAIN_DIR = r"D:\codeArea\work\ros_ws(1)\train_main\high-voltage cable.yolov8\train"
# =================================================

# 高压电线的 class_id
TARGET_CLASS_ID = "2"

def main():
    labels_dir = os.path.join(TRAIN_DIR, "labels")
    images_dir = os.path.join(TRAIN_DIR, "images")

    if not os.path.exists(labels_dir):
        print(f"[错误] labels 目录不存在: {labels_dir}")
        return
    if not os.path.exists(images_dir):
        print(f"[错误] images 目录不存在: {images_dir}")
        return

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    print(f"共找到 {len(txt_files)} 个标注文件")

    deleted_count = 0
    kept_count = 0

    for txt_name in txt_files:
        txt_path = os.path.join(labels_dir, txt_name)
        stem = os.path.splitext(txt_name)[0]  # 文件名（不含后缀）

        # 读取标注文件
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 只保留 class_id 为高压电线的行，保持原始顺序
        kept_lines = [line for line in lines if line.strip().split()[0] == TARGET_CLASS_ID]

        if len(kept_lines) == 0:
            # 没有高压电线标注，删除 txt 和对应图片
            os.remove(txt_path)

            # 查找对应图片（不管后缀是什么）
            image_deleted = False
            for img_name in os.listdir(images_dir):
                img_stem = os.path.splitext(img_name)[0]
                if img_stem == stem:
                    img_path = os.path.join(images_dir, img_name)
                    os.remove(img_path)
                    image_deleted = True
                    break

            if image_deleted:
                print(f"[删除] {txt_name} + 对应图片")
            else:
                print(f"[删除] {txt_name} (未找到对应图片)")

            deleted_count += 1
        else:
            # 有高压电线标注，只保留这些行，覆盖写回
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            kept_count += 1
            removed_lines = len(lines) - len(kept_lines)
            if removed_lines > 0:
                print(f"[保留] {txt_name} — 删除了 {removed_lines} 行非高压电线标注")

    print(f"\n完成！保留 {kept_count} 组，删除 {deleted_count} 组")


if __name__ == "__main__":
    main()