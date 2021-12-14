import pickle

import torch
from torch.nn import functional as F
import torch.optim as optim
from xml.parsers.expat import model

from main import SimpleNet
import main


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to('cpu')
            target = targets.to('cpu')
            # print(inputs.shape)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
        train_loss /= len(train_loader)
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to('cpu')
            output = model(inputs)
            targets = targets.to('cpu')
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, train_loss, valid_loss, num_correct / num_examples))

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
    #
    # test(dunw_net(), main.test_data_loader)
