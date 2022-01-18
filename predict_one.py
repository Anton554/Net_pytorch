import model_cnn
import cv2 as cv
import ansamble
import model_cnn
import torch
from torch.nn import functional as F


def print_proc(proc):
    for num, el in enumerate(proc, start=0):
        print(f'{num} - ({round(el, 3)})')
    print()


def predict(net, img_name: str):
    np_arr = cv.imread(img_name)  # shape [28, 28, 3]
    transform = model_cnn.get_transform()
    tr = transform(np_arr)
    input = tr.reshape(1, 3, 28, 28)
    pred = net(input)
    # pred = F.softmax(pred) * 100
    proc = [tn.item() for tn in pred[0]]
    pred = pred.argmax()
    return pred.item(), proc


if __name__ == '__main__':
    # net = model_cnn.CNNNet()
    # net.load_state_dict(torch.load('./net/cnn_net.pth'))
    net = ansamble.AnNNet()
    net.load_state_dict(torch.load('./net/ans_net.pth'))

    pred, proc = predict(net, './img/test/8/num-82.jpg')
    print_proc(proc)
    print(f'Ваше число - {pred}')
