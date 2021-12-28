import torch
from PIL import Image
import model
import os

lebles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def predict(net, path):
    """

    :param model: модель
    :param path: картинка - БЕЛАЯ НА ЧЁРНОМ ФОНЕ
    :return:
    """
    img_pil = Image.open(path)
    # Прогоняем через ТРАНСФОРМЕР
    transform = model.get_transform()
    tr = transform(img_pil)
    # Добавляем измерение пакета
    inputs = tr.resize_(1,3,28,28)
    print(f'{inputs.shape=}')
    # !!! Для отладки
    # return
    # Предсказаниие
    pred = net(inputs)
    pred =pred.argmax()
    print(f'{pred.shape=}')
    print(f'{pred=}')
    print(f'Предсказание: {pred.item()}')
    return pred.item()


def predict_folder(path:str, sub_path:str):
    """ Предсказание ВСЕХ картинок в папке

    :param path: путь к папке
    :param sub_path: путь к подпапке
    :return: None
    """
    # Загрузка модели
    net = torch.load('./cnn_net.pth')
    ls_img = os.listdir(path+ os.sep + sub_path)
    cn = 0
    for img in ls_img:
        img = path + os.sep + sub_path + os.sep + img
        print(f'{img=}')
        pred = predict(net, img)
        if pred == int(sub_path):
            cn += 1
        print('------------------')
    prc = round((cn / len(ls_img)) * 100, 2)
    print('========================')
    print(f'Процент предсказаний {prc} %')


if __name__ == '__main__':
    predict_folder('./img/td/', '7')