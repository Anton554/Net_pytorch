import sys

import numpy as np
import cv2 as cv

# параметры цветового фильтра
from matplotlib import pyplot as plt

width, height, mx = 0, 0, 0
x, y = 0, 0
hsv_ch = np.array((0, 0, 0), np.uint8)

if __name__ == '__main__':
    # путь к файлу с картинкой

    fn = r'C:\Projects\IT\Python\Net_pytorch\img\prov\Iz1_qUL4JWM.jpg'

    # преобразуем черно-белое изображение
    im_gray = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    # выделяет нужную информацию на картинке
    im_gray = cv.threshold(im_gray, 120, 255, cv.THRESH_BINARY_INV)[1]

    cv.imwrite('img_gr.jpg', im_gray)
    img_new = cv.imread('img_gr.jpg')
    # меняем цветовую модель с BGR на HSV
    hsv = cv.cvtColor(img_new, cv.COLOR_BGR2HSV)
    # применяем цветовой фильтр
    thresh = cv.inRange(hsv, hsv_ch, hsv_ch)
    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    ln = sorted([len(el) for el in contours])[-1]
    for el in contours:
        if len(el) == ln:
            # cv.drawContours(img_new, [el], 0, (0, 255, 0), 3)
            x = sorted([l[0][0] for l in el])[0] - 35
            y = sorted([l[0][1] for l in el])[0] - 35
            width = sorted([l[0][0] for l in el])[-1] - x + 35
            height = sorted([l[0][1] for l in el])[-1] - y + 35
            mx = max(width, height)
            break

    # Обрезаем изображение
    crop_img = img_new[y:y + mx, x:x + mx]

    morph_kernel = np.ones((6, 6), np.uint8)
    # Утолщаем изображение
    # erosion = cv.erode(crop_img, morph_kernel, iterations=1)
    # Утоньшаем изображение
    erosion = cv.dilate(crop_img, morph_kernel, iterations=1)
    erosion = cv.dilate(erosion, morph_kernel, iterations=1)
    # Выводим изображение
    # cv.imshow('EROSION', erosion)



    # Выводим итоговое изображение в окно
    cv.imshow('contours', erosion)
    cv.imwrite('crop_img.png', erosion)
    im_gray = cv.imread('crop_img.png', cv.IMREAD_GRAYSCALE)
    cv.imwrite('crop_img.png', im_gray)
    plt.show()
    cv.waitKey()
    cv.destroyAllWindows()
