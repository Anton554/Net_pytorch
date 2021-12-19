import pickle

import torch
from torch.nn import functional as F
import torch.optim as optim
from xml.parsers.expat import model

from main import SimpleNet
import main


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=10):
    """

    :param model: объект класса SimpleNet - модель нейронной сети
    :param optimizer: объект оптимизатор с алгоритмом "Адам"
    :param loss_fn: объект "функция ошибок" - подкласс CrossEntropyLoss
    :param train_loader: набор тренировочных данных shape(64, 3, 28, 28)
    :param val_loader: набор проверочных данных shape(64, 3, 28, 28)
    :param epochs: кол-во эпох
    :return:
    """
    for epoch in range(epochs):
        # Обнуление ТРЕНИРОВОЧНОГО и ПРОВЕРОЧНОГО счетчика ошибок
        train_loss = 0.0
        valid_loss = 0.0
        # Включение ОБУЧАЮЩЕГО режима у модели
        model.train()
        for batch in train_loader:
            # После каждого батча обнуляем градиент
            optimizer.zero_grad()
            # inputs - Tensor (64, 3, 28, 38) - картинки для обучения
            # targets - правильный ответ Tensor(64,)
            inputs, targets = batch
            # копируем Tensor в память cpu
            inputs = inputs.to('cpu')
            target = targets.to('cpu')
            # Предсказываем значаение ОБУЧАЮЩЕГО картинки
            output = model(inputs)
            # вычисление значения "функции потерь"
            loss = loss_fn(output, target)
            # вычисление градиента "функции потерь" (вектор направление)
            loss.backward()
            # обновление градиента "функции потерь" обратное распространение ошибки
            optimizer.step()
            # накапливаем значение "функции потерь"
            train_loss += loss.data.item()
        # для каждой эпохи находим среднее арифмитическое значение "функции потерь"
        # в итоге получаем среднее значение функции ошибки ТРЕНИРОВОЧНОГО пакета
        train_loss /= len(train_loader)
        # Включение ПРОВЕРОЧНОГО режима у модели
        model.eval()
        # обнуление счетчиков
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            # inputs - Tensor (64, 3, 28, 38) - картинки для ПРОВЕРКИ
            # targets - правильный ответ Tensor(64,)
            inputs, targets = batch
            # копируем Tensor в память cpu
            inputs = inputs.to('cpu')
            targets = targets.to('cpu')
            # Предсказываем значаение ПРОВЕРОЧНОЙ картинки
            output = model(inputs)
            # вычисление значения "функции потерь"
            loss = loss_fn(output, targets)
            # накапливаем значение "функции потерь"
            valid_loss += loss.data.item()
            """1) softmax мерному входному тензору, масштабируя их так, чтобы элементы n-мерного выходного тензора 
            находились в диапазоне [0,1] 
            2) max возвращает namedtuple(values, indices), где values максимальное значение каждой строки inputтензора
             в данном измерении dim
            3) eq вычисляет поэлементное равнство
            4) получаем тензор tensor([ True,  True,  True,  True])"""
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            # кол-во правильных ответов
            num_correct += torch.sum(correct).item()
            # накапливаем общее кол-во проверок
            num_examples += correct.shape[0]
        # для каждой эпохи находим среднее арифмитическое значение "функции потерь"
        # в итоге получаем среднее значение функции ошибки ПРОВЕРОЧНОГО пакета
        valid_loss /= len(val_loader)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, train_loss,
                                                                                                    valid_loss,
                                                                                                    num_correct / num_examples))


def dunw_net():
    with open('simplenet.pth', 'rb') as f:
        simplenet = pickle.load(f)
    return simplenet


def test(model, test_loader):
    # Load the model that we saved at the end of the training loop
    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            print(outputs.shape)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print('Accuracy of the model based on the test set of '
              'inputs is: %d %%' % (100 * running_accuracy / total))


if __name__ == "__main__":
    simplenet = SimpleNet()
    optimizator = optim.Adam(simplenet.parameters(), lr=0.001)
    train(simplenet, optimizator, torch.nn.CrossEntropyLoss(), main.train_data_loader, main.val_data_loader)
    with open('simplenet.pth', 'wb') as f:
        pickle.dump(simplenet, f)
