"""
Проверка сетина на ТЕСТОВОМ наборе картинок
"""
import torch
import model_cnn


def testing_model(model, test_loader):
    cn_true = 0
    for batch in test_loader:
        inputs, targets = batch
        print(f'{inputs.shape=}')
        targets = targets[0]
        print(f'{targets.shape=}')
        print(f'{targets=}')

        pred = model(inputs)
        print(f'{pred.shape=}')

        pred = pred.argmax()
        # print(f'{torch.eq(pred, targets).item()=}')
        if torch.eq(pred, targets).item():
            cn_true += 1
        print(f'{pred=}')
        print('===============================')
    print(f'Процент предсказаний на {len(test_loader)} изображений = {round((cn_true/len(test_loader)*100),2)} %')

if __name__ == '__main__':
    # Создание загрузчика
    loader = model_cnn.DtLoader('./img/train', './img/val', './img/test')
    # Загрузка модели
    save_model = torch.load('./cnn_net.pth')
    # Тестирование модели
    testing_model(save_model, loader.test_data_loader)
    # print('===============================')
    # print('Архитектура сети')
    # print('--------------------------------')
    # print(save_model)
