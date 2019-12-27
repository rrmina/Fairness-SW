import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset():
    r = np.random.uniform(0, 1, 100000)
    r = (r > 0.5)*1
    r = np.expand_dims(r, 1)

    v = np.random.normal(r, 1)
    u = np.random.normal(v, 1)
    w = np.random.normal(v, 1)

    x = np.hstack((r, u))
    y = (w > 0) * 1

    return x, y, r

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(2, 1)
        #self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.sigmoid1(x)
        return x

class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()

        self.fc1 = nn.Linear(1, 1)
        #self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.sigmoid1(x)
        return x

def bce_loss(out, label):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(out, label.float())

# HYPERPARAMETER SETTINGS
LR = 0.2
NUM_EPOCHS = 5000

# MODE
ADVERSARIAL_MODE = 1

def main():
    # Seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # Load dataset
    xnp, ynp, znp = generate_dataset()
    x = torch.from_numpy(xnp)
    y = torch.from_numpy(ynp)
    z = torch.from_numpy(znp)

    # Load Networks
    p = Predictor().float()
    a = Adversary().float()

    # Optimizer
    p_optim = optim.SGD(p.parameters(), lr=LR)
    a_optim = optim.SGD(a.parameters(), lr=LR)

    # Training Loop
    pred_loss_hist = []
    pred_acc_hist = []
    adver_loss_hist = []
    adver_acc_hist = []
    for i in range(1, NUM_EPOCHS + 1):

        if (ADVERSARIAL_MODE):
            # Adversary
            a_optim.zero_grad()
            adver_in = torch.sigmoid(p(x.float()))
            a_out = a(adver_in)
            a_loss = bce_loss(a_out, z)
            adver_loss_hist.append(a_loss.item())
            a_loss.backward()
            a_optim.step()

            # Compute Adversary accuracy
            adver_acc = np.mean(((torch.sigmoid(a_out).detach().numpy()) > 0.5) == znp)
            adver_acc_hist.append(adver_acc)


        # Predictor
        p_optim.zero_grad()
        p_out = p(x.float())
        p_loss = bce_loss(p_out, y)
        pred_loss_hist.append(p_loss.item())

        # Predictor -> Adversary
        if (ADVERSARIAL_MODE):
            pa_out = a(p_out)
            pa_loss = bce_loss(pa_out, z)

            total_ploss = p_loss - pa_loss

        else:
            total_ploss = p_loss
        
        total_ploss.backward()
        p_optim.step()

        # Compute Predictor accuracy
        acc = np.mean(((torch.sigmoid(p_out).detach().numpy()) > 0.5) == ynp)
        pred_acc_hist.append(acc)


    # Print the parameter values
    
    i = 0
    for param in p.parameters():
        if param.requires_grad:
            if (i == 0):
                print("Weight of r (senstive attribute), and weight of u {}".format(param.data))
            else:
                print("Bias (parameter) value {}".format(param.data))



    # Predictor Plots
    plt.plot(pred_loss_hist)
    plt.title("Predictor Loss vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Predictor Loss")
    plt.show()
    plt.close()
    plt.plot(pred_acc_hist)
    plt.title("Predictor Accuracy vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Predictor Accuracy")
    plt.show()
    plt.close()

    # Adversary Plots
    if (ADVERSARIAL_MODE):
        plt.plot(adver_loss_hist)
        plt.title("Adversary Loss vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Adversary Loss")
        plt.show()
        plt.close()
        plt.plot(adver_acc_hist)
        plt.title("Adversary Accuracy vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Adversary Accuracy")
        plt.show()
        plt.close()

main()