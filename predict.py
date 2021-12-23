import torch
from torch.nn import functional as F
from PIL import Image
import model
import os

lebles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def predict(path):
    """

    :param model: модель
    :param path: картинка - БЕЛАЯ НА ЧЁРНОМ ФОНЕ
    :return:
    """
    img_pil = Image.open(path)
    net = torch.load('./simplenet9667.pth')
    # Прогоняем через ТРАНСФОРМЕР
    transform = model.get_transform()
    tr = transform(img_pil)
    # Добавляем измерение пакета
    inputs = tr.reshape(1, 1, 28, 28)
    # Предсказаниие
    pred = net(inputs)
    pred = F.softmax(pred, dim=1).argmax()
    print(f'Предсказание: {pred.item()}')
    return pred.item()


if __name__ == '__main__':
    predict(r'C:\Projects\IT\Python\PyTorch\Test\crop_img.png')

