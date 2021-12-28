import torch
import model_cnn
import cv2 as cv


def predict(net, img_name:str):
    np_arr = cv.imread(img_name)  # shape [28, 28, 3]
    print(f'{np_arr.shape=}')
    transform = model_cnn.get_transform()
    tr = transform(np_arr)
    input = tr.reshape(1,3,28,28)
    print(f'{input.shape=}')

    pred = net(input)
    print(f'{pred.shape=}')
    pred = pred.argmax()
    return pred.item()


if __name__ == '__main__':
    net = torch.load('./net/cnn_net_3ch.pth')
    pred = predict(net, './img/td/9/num-11.jpg')
    print(pred)
