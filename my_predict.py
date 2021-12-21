import torch
from torch.nn import functional as F
from PIL import Image
import model

lebles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def my_predict(net, path):
    """

    :param model: модель
    :param path: картинка - БЕЛАЯ НА ЧЁРНОМ ФОНЕ
    :return:
    """
    img_pil = Image.open(path)
    # Прогоняем через ТРАНСФОРМЕР
    transform = model.get_transform()
    tr = transform(img_pil)
    # tr = 255 - tr
    print(f'{tr.shape=}')
    # !!! Для отладки
    # return
    # Добавляем измерение пакета
    inputs = tr.reshape(1,1,28,28)
    print(f'{inputs.shape=}')
    # Предсказаниие
    pred = net(inputs)
    pred = F.softmax(pred, dim=1).argmax()
    print(f'{pred.shape=}')
    print(f'{pred=}')
    print(f'Предсказание: {pred.item()}')

if __name__ == '__main__':
    # Загрузка модели
    save_model = torch.load('./simplenet9667.pth')
    # my_predict(save_model, './img/predict/5.jpg')
    my_predict(save_model, './img/test/5/num-162.jpg')