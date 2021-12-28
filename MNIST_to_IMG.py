"""
Преобразует набор данных MNIST из матрицы в файл картинок
"""
import pandas as pd
import numpy as np
import cv2 as cv



def read_df(path):
    """ Загрузка csv в DataFrame

    :param path: Путь к файлу вида './train.csv'
    :return: DataFrame
    """
    df = pd.read_csv(path, sep=',', header=0)
    print(f'Файл {path} прочитан')

    cn = len(df)
    ls_all = []
    for i in range(cn):
        ls = df.iloc[i].to_list()
        ls_all.append([ls[0], tuple(ls[1:])])
    return ls_all


def save2folder(ls, typ_dir):
    """ Преобразут DataFrame в картинки и делит по каталогам

    :param df: DataFrame изображений
    :param typ_dir: Тип каталога train/val
    :return: None
    """
    cn = len(ls)
    for i in range(cn):
        y = ls[i][0]
        # нормализация
        # np_img = np_img/255
        # Инвертирование
        np_img = np.array(ls[i][1])
        # np_img = 255 - np_img.reshape(28,28)
        np_img = np_img.reshape(28,28)
        print(f'{i=} из {cn}')
        cv.imwrite(f'./img/{typ_dir}/{y}/num-{i}.jpg', np_img)



if __name__ == '__main__':
    ls = read_df('./train.csv')
    cn_test = int(len(ls)*0.01 //1)
    cn_val = int(len(ls)*0.29 //1)
    cn_train = int(len(ls)*0.70 //1)

    print(cn_test, cn_val, cn_train)

    save2folder(ls[:cn_test], 'test')
    save2folder(ls[cn_test:cn_val], 'val')
    save2folder(ls[cn_val:], 'train')

