import cv2
import numpy as np
from PIL import Image
import pytesseract


def img_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h的值
    widthA = np.sqrt((((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)))
    widthB = np.sqrt((((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)))
    heightB = np.sqrt((((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应的坐标位置
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


img = cv2.imread('pos.jpg')
ratio = img.shape[0] / 500.0
orig = img.copy()

img = resize(orig, height = 500)

# 先拿到边缘信息
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# STEP 1 边缘检测结果
print('======STEP 1 边缘检测======')
img_show('edged', edged)

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # 拿到所有的轮廓
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[: 5]  # 按照矩形大小排序获取信息 拿到前五个大的 为了方便多个纸片的情况

# 遍历轮廓
for c in cnts:
    # 计算轮廓长度
    peri = cv2.arcLength(c, True)  # 计算长度
    # 计算一下轮廓近似 近似出来轮廓的矩形 True 表示封闭
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    # 四个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("======STEP 2 获取轮廓======")
cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
img_show('CNT', img)

# 做透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值化
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("res.jpg", ref)
# 展示结果
print("======STEP 3 透视变换======")
cv2.imshow("pos", resize(orig, height=650))
cv2.imshow("res", resize(ref, height=650))


image = cv2.imread("res.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
preprocess = 'thresh' #thresh  #做预处理选项
if preprocess == 'blur':
    gray = cv2.blur(gray,(3,3))
if preprocess == 'thresh':
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)


text = pytesseract.image_to_string(Image.open("res.jpg"))  # 转化成文本
print("======STEP 4 OCR识别======")
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()

