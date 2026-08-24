import cv2

conf = cv2.imread(
    r"output/infer_HQL/HQL_0730/D_7_confidence/01-01_confidence.png",
    cv2.IMREAD_UNCHANGED,
)

print(conf.dtype)
print(conf.min(), conf.max())