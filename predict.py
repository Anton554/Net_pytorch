import pickle
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from main import trans
lebles = ['1', '0']


def my_predict(path):
    img = Image.open(path)

    resized_img = cv2.resize(np.asarray(img), (28,28), interpolation=cv2.INTER_AREA)

    # img = Image.fromarray(np.uint8(resized_img)).convert('RGB')
    # print(resized_img.shape)
    thresh = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    thresh = np.expand_dims(thresh, axis = 0)
    print(thresh.shape)
    # plt.imshow(thresh)
    # plt.show()
    demo_img = Image.fromarray(np.uint8(resized_img)).convert('RGB')
    demo_img = trans(demo_img).permute(1, 2, 0)
    demo_img = Image.fromarray(np.uint8(demo_img)).convert('RGB')
    # print(type(demo_img))

    demo_img = trans(demo_img)
    img = demo_img.unsqueeze(0)
    with open('simplenet.pth', 'rb') as f:
        simplenet = pickle.load(f)
    pred = simplenet(img)
    pred = pred.argmax()
    return lebles[pred]


if __name__ == '__main__':
    # print(my_predict('crop_img.png'))
    # print(my_predict(r'C:\Users\админ\Downloads\QIsGWFw-nQg.jpg'))
    # print(my_predict(r'C:\Users\админ\Downloads\Ta184KvmO3s.jpg'))
    print(my_predict(r'C:\Projects\IT\Python\PyTorch\Mesh_network\img\test\0\num-5560.jpg'))
    print(my_predict(r'C:\Projects\IT\Python\PyTorch\Mesh_network\img\test\1\num-5265.jpg'))
    print(my_predict(r'C:\Projects\IT\Python\PyTorch\Mesh_network\img\test\0\num-5577.jpg'))
    print(my_predict(r'C:\Projects\IT\Python\PyTorch\Mesh_network\img\test\1\num-5266.jpg'))