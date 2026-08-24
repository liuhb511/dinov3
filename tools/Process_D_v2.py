import cv2 as cv
import json
import numpy as np
from skimage.draw import draw
import os


class Process:
    def __init__(self):
        self.path = ""
        self.imgs_path = []
        self.save_path = ""

        # 只统计D、HC、SZ三个类别
        self.JZW_COUNT = [0, 0, 0]
        self.img_COUNT = [0, 0, 0]

        # 0为背景，1～3为有效类别
        self.label_map = {
            "D": 1,
            "HC": 2,
            "SZ": 3,
        }

    def loadjson(self, path, savepath):
        self.path = path
        self.imgs_path = os.listdir(path)
        self.save_path = savepath

    def run(self, cropType):
        for imgpath in self.imgs_path:
            if not imgpath.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(self.path, imgpath)
            json_path = os.path.join(
                self.path,
                os.path.splitext(imgpath)[0] + ".json"
            )

            if not os.path.exists(json_path):
                print("找不到对应JSON，已跳过：", json_path)
                continue

            img = cv.imread(image_path)

            if img is None:
                print("图片读取失败，已跳过：", image_path)
                continue

            mask = self.json2voc(imgpath)

            # 只保存包含D/HC/SZ有效区域的图片
            if np.max(mask) > 0:
                self.enhance(img, mask, imgpath)
                print(imgpath + " ---- Save")
                print("----------------------")
            else:
                print(imgpath + " ---- 不包含D/HC/SZ，已跳过")
                print("----------------------")

        # 需要裁剪时取消下面这一行的注释
        # self.crop(cropType)

    def json2voc(self, imgpath):
        image_path = os.path.join(self.path, imgpath)
        json_path = os.path.join(
            self.path,
            os.path.splitext(imgpath)[0] + ".json"
        )

        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("图片读取失败：" + image_path)

        # mask初始全部为0，即背景
        mask = np.zeros(img.shape, dtype=np.uint8)

        # 记录当前图片中是否出现D、HC、SZ
        jzw_flag = [0, 0, 0]

        with open(json_path, encoding="utf-8") as file:
            result = json.load(file)
            shapes = result.get("shapes", [])

            for shape in shapes:
                label = str(shape.get("label", "")).strip().upper()

                # 除D/HC/SZ之外的类别全部作为背景
                # 因为mask初始值就是0，所以直接跳过即可
                if label not in self.label_map:
                    continue

                class_id = self.label_map[label]
                points = shape.get("points", [])
                shape_type = shape.get("shape_type", "polygon")

                # ---- D/HC/SZ 均支持圆形标注 ----
                if shape_type == "circle":
                    if len(points) < 2:
                        print(
                            "圆形标注点数量不足（需要2个点：圆心+圆周），已跳过：",
                            imgpath
                        )
                        continue
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    center = (int(round(x1)), int(round(y1)))
                    radius = int(
                        ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    ) + 1
                    cv.circle(mask, center, radius, class_id, -1)
                    self.JZW_COUNT[class_id - 1] += 1
                    jzw_flag[class_id - 1] = 1
                    continue

                # ---- 多边形路径（D/HC/SZ 通用） ----
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

                    # skimage.draw.polygon的顺序是行、列，即y、x
                    rows.append(y)
                    cols.append(x)

                rr, cc = draw.polygon(
                    np.array(rows),
                    np.array(cols),
                    shape=mask.shape
                )

                # 写入类别值
                draw.set_color(mask, (rr, cc), class_id)

                # 目标数量统计
                self.JZW_COUNT[class_id - 1] += 1

                # 当前图片包含该类别
                jzw_flag[class_id - 1] = 1

        # 图片数量统计
        self.img_COUNT = (
            np.array(self.img_COUNT, dtype=np.int64)
            + np.array(jzw_flag, dtype=np.int64)
        ).tolist()

        return mask

    def enhance(self, img, mask, name):
        total_path = os.path.join(self.save_path, "total")
        jpeg_path = os.path.join(total_path, "JPEGImages")
        segmentation_path = os.path.join(
            total_path,
            "SegmentationClass"
        )

        # 自动创建目录
        os.makedirs(jpeg_path, exist_ok=True)
        os.makedirs(segmentation_path, exist_ok=True)

        image_save_path = os.path.join(jpeg_path, name)

        mask_name = os.path.splitext(name)[0] + ".png"
        mask_save_path = os.path.join(
            segmentation_path,
            mask_name
        )

        image_result = cv.imwrite(image_save_path, img)
        mask_result = cv.imwrite(mask_save_path, mask)

        if not image_result:
            print("原图保存失败：", image_save_path)

        if not mask_result:
            print("标签保存失败：", mask_save_path)

        # ---------------- 数据增强预留 ----------------

        # 旋转90度
        # img90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        # mask90 = cv.rotate(mask, cv.ROTATE_90_CLOCKWISE)
        #
        # cv.imwrite(
        #     os.path.join(
        #         jpeg_path,
        #         os.path.splitext(name)[0] + "-rotate-90.jpg"
        #     ),
        #     img90
        # )
        #
        # cv.imwrite(
        #     os.path.join(
        #         segmentation_path,
        #         os.path.splitext(name)[0] + "-rotate-90.png"
        #     ),
        #     mask90
        # )

        # 垂直翻转
        # img_flip = cv.flip(img, 0)
        # mask_flip = cv.flip(mask, 0)

    def crop(self, crop_type):
        crop_size = 512

        crop_path = os.path.join(self.save_path, "crop")
        jpeg_crop_path = os.path.join(
            crop_path,
            "JPEGImages"
        )
        segmentation_crop_path = os.path.join(
            crop_path,
            "SegmentationClass"
        )

        os.makedirs(jpeg_crop_path, exist_ok=True)
        os.makedirs(segmentation_crop_path, exist_ok=True)

        image_total_path = os.path.join(
            self.save_path,
            "total",
            "JPEGImages"
        )
        mask_total_path = os.path.join(
            self.save_path,
            "total",
            "SegmentationClass"
        )

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
            mask_path = os.path.join(
                mask_total_path,
                os.path.splitext(item)[0] + ".png"
            )

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

            # 使用向上取整，保证边缘区域也会被处理
            count_h = int(np.ceil(height / crop_size))
            count_w = int(np.ceil(width / crop_size))

            for row_index in range(count_h):
                for col_index in range(count_w):
                    y1 = row_index * crop_size
                    x1 = col_index * crop_size

                    y2 = min(y1 + crop_size, height)
                    x2 = min(x1 + crop_size, width)

                    # 如果位于边缘，使裁剪区域尽量保持512×512
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

                    image_crop_path = os.path.join(
                        jpeg_crop_path,
                        crop_name + ".jpg"
                    )

                    mask_crop_path = os.path.join(
                        segmentation_crop_path,
                        crop_name + ".png"
                    )

                    if crop_type == "nobg":
                        # 只保存含D/HC/SZ的图块
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
        print("D类：", self.JZW_COUNT[0], "个")
        print("HC：", self.JZW_COUNT[1], "个")
        print("SZ：", self.JZW_COUNT[2], "个")

        print("-----------图片数量统计-----------")
        print("D类：", self.img_COUNT[0], "张")
        print("HC：", self.img_COUNT[1], "张")
        print("SZ：", self.img_COUNT[2], "张")

        print("-----------标签对应关系-----------")
        print("背景：0")
        print("D类：1")
        print("HC：2")
        print("SZ：3")


if __name__ == "__main__":
    path = "F:/liuhaibo/datasets/JZW_v3/crop_1024/images/"
    savepath = "F:/liuhaibo/datasets/JZW_v3/crop_1024/D_1024/"

    processor = Process()
    processor.loadjson(path, savepath)

    # nobg表示裁剪时不保存纯背景图块
    processor.run("nobg")
    processor.info()

        #
        # cv.imwrite(
        #     os.path.join(
        #         jpeg_path,
        #         os.path.splitext(name)[0] + "-flip.jpg"
        #     ),
        #     img_flip
        # )
        #
        # cv.imwrite(
        #     os.path.join(
        #         segmentation_path,
        #         os.path.splitext(name)[0] + "-flip.png"
        #     ),
        #     mask_flip
        # )
