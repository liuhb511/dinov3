import cv2 as cv
import json
import numpy as np
from skimage.draw import draw
import os


class Process:
    def __init__(self, classes):
        """
        动态类别 + 任意类别合并

        规则：
        1. 背景固定为 0。
        2. classes 数组的“顶层元素”顺序决定类别索引。
        3. 单个字符串表示独立类别。
        4. list / tuple / set 表示这些类别合并成同一个类别。
        5. 类别索引 = 顶层数组下标 + 1。

        示例：
            classes = [
                "A",
                "B",
                ["TIN-B", "TIN-C"],
                "TIN-D"
            ]

        对应：
            背景            -> 0
            A               -> 1
            B               -> 2
            TIN-B / TIN-C   -> 3
            TIN-D           -> 4

        JSON 中的 TIN-B 和 TIN-C 都会被写成 mask 值 3。
        """

        self.path = ""
        self.imgs_path = []
        self.save_path = ""

        # 每个索引对应的“类别组”
        # 例如：
        # [
        #   ["A"],
        #   ["B"],
        #   ["TIN-B", "TIN-C"],
        #   ["TIN-D"]
        # ]
        self.class_groups = []

        for item in classes:
            # 合并类别组
            if isinstance(item, (list, tuple, set)):
                group = [
                    str(label).strip().upper()
                    for label in item
                    if str(label).strip()
                ]
            else:
                # 单独类别
                label = str(item).strip().upper()
                group = [label] if label else []

            if not group:
                raise ValueError(
                    "classes 中存在空类别或空类别组，请检查：{}".format(item)
                )

            # 同一个合并组内部不允许重复
            if len(group) != len(set(group)):
                raise ValueError(
                    "同一个类别组中存在重复类别：{}".format(group)
                )

            self.class_groups.append(group)

        if not self.class_groups:
            raise ValueError("classes 不能为空")

        # -------------------------------------------------
        # 建立 label -> class_id 映射
        #
        # 顶层第 0 个元素 -> 1
        # 顶层第 1 个元素 -> 2
        # ...
        #
        # 如果某个顶层元素是 ["TIN-B", "TIN-C"]，
        # 那么 TIN-B 和 TIN-C 会映射到相同 class_id。
        # -------------------------------------------------
        self.label_map = {}

        for class_index, group in enumerate(self.class_groups):
            class_id = class_index + 1

            for label in group:
                # 防止一个原始类别同时出现在多个组中
                if label in self.label_map:
                    raise ValueError(
                        "类别 {} 被重复配置到了多个类别组中".format(label)
                    )

                self.label_map[label] = class_id

        # 每个“最终类别”的显示名称
        # ["TIN-B", "TIN-C"] -> "TIN-B/TIN-C"
        self.class_names = ["/".join(group) for group in self.class_groups]

        # 最终类别数 = 顶层数组长度
        class_count = len(self.class_groups)

        # 目标数量统计
        self.JZW_COUNT = [0] * class_count

        # 图片数量统计
        self.img_COUNT = [0] * class_count

        print("-----------当前类别对应关系-----------")
        print("背景：0")

        for class_id, group in enumerate(self.class_groups, start=1):
            if len(group) == 1:
                print("{}：{}".format(group[0], class_id))
            else:
                print("{}：{}  （合并类别）".format(
                    "/".join(group),
                    class_id
                ))

                # 同时打印组内各原始标签对应关系，方便检查
                for label in group:
                    print("    {} -> {}".format(label, class_id))

        print("------------------------------------")

    def loadjson(self, path, savepath):
        self.path = path
        self.imgs_path = os.listdir(path)
        self.save_path = savepath

    def run(self, cropType):
        for imgpath in self.imgs_path:
            if not imgpath.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(self.path, imgpath)
            json_path = os.path.join(self.path, os.path.splitext(imgpath)[0] + ".json")

            if not os.path.exists(json_path):
                print("找不到对应JSON，已跳过：", json_path)
                continue

            img = cv.imread(image_path)

            if img is None:
                print("图片读取失败，已跳过：", image_path)
                continue

            mask = self.json2voc(imgpath)

            # 只保存至少包含一个指定类别有效区域的图片
            if np.max(mask) > 0:
                self.enhance(img, mask, imgpath)
                print(imgpath + " ---- Save")
                print("----------------------")
            else:
                print(
                    imgpath
                    + " ---- 不包含指定类别 "
                    + str(self.class_names)
                    + "，已跳过"
                )
                print("----------------------")

        # 需要裁剪时取消下面这一行的注释
        # self.crop(cropType)

    def json2voc(self, imgpath):
        image_path = os.path.join(self.path, imgpath)
        json_path = os.path.join(self.path, os.path.splitext(imgpath)[0] + ".json")

        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("图片读取失败：" + image_path)

        # mask 初始全部为 0，即背景
        mask = np.zeros(img.shape, dtype=np.uint8)

        # 每个“最终类别”在当前图片中是否出现
        jzw_flag = [0] * len(self.class_groups)

        with open(json_path, encoding="utf-8") as file:
            result = json.load(file)
            shapes = result.get("shapes", [])

            for shape in shapes:
                label = str(shape.get("label", "")).strip().upper()

                # 不在配置中的类别全部作为背景
                if label not in self.label_map:
                    continue

                points = shape.get("points", [])

                # 多边形至少需要 3 个点
                if len(points) < 3:
                    print(
                        "标注点数量不足，已跳过：",
                        imgpath,
                        label,
                        points
                    )
                    continue

                rows = []
                cols = []

                for point in points:
                    x = int(round(point[0]))
                    y = int(round(point[1]))

                    # skimage.draw.polygon 的顺序是 行、列，即 y、x
                    rows.append(y)
                    cols.append(x)

                rr, cc = draw.polygon(np.array(rows), np.array(cols), shape=mask.shape)

                # 合并组中的类别会得到相同的 class_id
                class_id = self.label_map[label]

                # 写入类别值
                draw.set_color(mask, (rr, cc), class_id)

                # -------------------------------------------------
                # 目标数量统计
                #
                # 例如 TIN-B 和 TIN-C 合并为 class_id=3，
                # 那么两者的实例都会累加到 JZW_COUNT[2]。
                # -------------------------------------------------
                self.JZW_COUNT[class_id - 1] += 1

                # 当前图片包含这个“最终类别”
                # 即使同一张图里同时有 TIN-B 和 TIN-C，
                # 图片数量也只会为这个合并类别增加 1。
                jzw_flag[class_id - 1] = 1

        # 图片数量统计
        self.img_COUNT = (np.array(self.img_COUNT, dtype=np.int64) + np.array(jzw_flag, dtype=np.int64)).tolist()

        return mask

    def enhance(self, img, mask, name):
        total_path = os.path.join(self.save_path, "total")
        jpeg_path = os.path.join(total_path, "JPEGImages")
        segmentation_path = os.path.join(total_path, "SegmentationClass")

        os.makedirs(jpeg_path, exist_ok=True)
        os.makedirs(segmentation_path, exist_ok=True)

        image_save_path = os.path.join(jpeg_path, name)

        mask_name = os.path.splitext(name)[0] + ".png"
        mask_save_path = os.path.join(segmentation_path, mask_name)

        image_result = cv.imwrite(image_save_path, img)
        mask_result = cv.imwrite(mask_save_path, mask)

        if not image_result:
            print("原图保存失败：", image_save_path)

        if not mask_result:
            print("标签保存失败：", mask_save_path)

    def crop(self, crop_type):
        crop_size = 512

        crop_path = os.path.join(self.save_path, "crop")
        jpeg_crop_path = os.path.join(crop_path, "JPEGImages")
        segmentation_crop_path = os.path.join(crop_path, "SegmentationClass")

        os.makedirs(jpeg_crop_path, exist_ok=True)
        os.makedirs(segmentation_crop_path, exist_ok=True)

        image_total_path = os.path.join(self.save_path, "total", "JPEGImages")
        mask_total_path = os.path.join(self.save_path, "total", "SegmentationClass")

        if not os.path.exists(image_total_path):
            print("原图目录不存在：", image_total_path)
            return

        if not os.path.exists(mask_total_path):
            print("标签目录不存在：", mask_total_path)
            return

        imgs = os.listdir(image_total_path)

        for item in imgs:
            if not item.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(image_total_path, item)
            mask_path = os.path.join(mask_total_path, os.path.splitext(item)[0] + ".png")

            if not os.path.exists(mask_path):
                print("找不到对应标签，已跳过：", mask_path)
                continue

            img = cv.imread(image_path)
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print("图片读取失败：", image_path)
                continue

            if mask is None:
                print("标签读取失败：", mask_path)
                continue

            height, width = img.shape[:2]

            count_h = int(np.ceil(height / crop_size))
            count_w = int(np.ceil(width / crop_size))

            for row_index in range(count_h):
                for col_index in range(count_w):
                    y1 = row_index * crop_size
                    x1 = col_index * crop_size

                    y2 = min(y1 + crop_size, height)
                    x2 = min(x1 + crop_size, width)

                    # 边缘区域尽量保持 512 × 512
                    y1 = max(0, y2 - crop_size)
                    x1 = max(0, x2 - crop_size)

                    img_crop = img[y1:y2, x1:x2]
                    mask_crop = mask[y1:y2, x1:x2]

                    crop_name = (
                        os.path.splitext(item)[0]
                        + "_"
                        + str(row_index)
                        + "_"
                        + str(col_index)
                    )

                    image_crop_path = os.path.join(jpeg_crop_path, crop_name + ".jpg")

                    mask_crop_path = os.path.join(segmentation_crop_path, crop_name + ".png")

                    if crop_type == "nobg":
                        # 只保存包含至少一个有效类别的图块
                        if np.max(mask_crop) > 0:
                            cv.imwrite(image_crop_path, img_crop)
                            cv.imwrite(mask_crop_path, mask_crop)
                    else:
                        # 保存所有图块，包括纯背景图块
                        cv.imwrite(image_crop_path, img_crop)
                        cv.imwrite(mask_crop_path, mask_crop)

            print(item + " ---- Crop")
            print("----------------------")

    def info(self):
        print("-----------目标数量统计-----------")
        for index, class_name in enumerate(self.class_names):
            print("{}类：{}个".format(class_name, self.JZW_COUNT[index]))

        print("-----------图片数量统计-----------")
        for index, class_name in enumerate(self.class_names):
            print("{}类：{}张".format(class_name, self.img_COUNT[index]))

        print("-----------标签对应关系-----------")
        print("背景：0")

        for class_id, group in enumerate(self.class_groups, start=1):
            print("{}：{}".format("/".join(group), class_id))


if __name__ == "__main__":
    path = r"D:/lhb/datasets/JZW_v4/process/JZW_V4_1024/images/"
    savepath = r"D:/lhb/datasets/JZW_v4/train/D/"
    """
    # ==========================================================
    # 只需要修改这里
    #
    # 【单独类别】
    # 写字符串：
    #
    # classes = [
    #     "A",
    #     "B",
    #     "C"
    # ]
    #
    # 结果：
    # 背景=0
    # A=1
    # B=2
    # C=3
    #
    #
    # 【合并类别】
    # 需要合并的类别放到同一个 list 中：
    #
    # classes = [
    #     "A",
    #     "B",
    #     ["TIN-B", "TIN-C"],
    #     "TIN-D"
    # ]
    #
    # 结果：
    # 背景=0
    # A=1
    # B=2
    # TIN-B=3
    # TIN-C=3
    # TIN-D=4
    #
    #
    # 【任意多个类别都可以合并】
    #
    # classes = [
    #     ["A", "B", "C"],
    #     ["TIN-B", "TIN-C", "TIN-D"]
    # ]
    #
    # 结果：
    # 背景=0
    # A/B/C=1
    # TIN-B/TIN-C/TIN-D=2
    # ==========================================================
    """
    classes = ["D"]
    # classes = ["A", "B", "C", ["TIN-B", "TIN-C", "TIN-B/TIN-C"], "TIN-D"]

    processor = Process(classes)
    processor.loadjson(path, savepath)

    # nobg 表示裁剪时不保存纯背景图块
    processor.run("nobg")
    processor.info()
