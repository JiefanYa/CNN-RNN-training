import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

Delta = (0, 2 * np.pi, 100)
IN = 10
OUT = 90
device = torch.device('cpu')
dtype = torch.float32
print_every = 50

input_dim = 10
embed_dim = 256
fc2_dim = 128
lstm_dim = 64
hidden_dim = 512
sequence_num = OUT
output_dim = 1
learning_rate = 4e-3

dir = os.path.dirname(__file__)
dset_dir = os.path.join(dir, './data')
model_path = os.path.join(dir, 'model.pt')

class SineWaveDataset(Dataset):

    def __init__(self, data_in, data_out, train):
        """
        Inputs:
        - data_in: Input sine waves of shape (x, IN)
        - data_out: Output sine waves of shape (x, OUT)
        - train: whether this is validation vs. test set
        """
        if data_in.shape[0] != data_out.shape[0]:
            raise Exception('data dimension conflict!')
        self.data_in = data_in
        self.data_out = data_out
        self.train = train

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_in[idx], self.data_out[idx]


def buildData(x, Delta, IN, OUT, type, dir=dset_dir):
    """
    Generates random sine waves.

    Input:
    - x: Number of sine waves produced
    - Delta: Linspace of the sine waves, number of separation = IN + OUT
    - IN: Number of input numbers in the interval defined by Delta
    - OUT: Number of output numbers in the interval defined by Delta

    Returns a tuple of:
    - input: Input sine waves of shape (x, IN)
    - output: Output sine waves of shape (x, OUT)
    """
    # frequency w
    frequency = np.random.randn(x) + 1
    frequency[frequency == 0] = 1

    frequency = np.ones_like(frequency)

    # amplitude A
    amplitude = (np.random.randn(x) + 1) * 2
    amplitude[amplitude == 0] = 1
    # phase Phi
    phase = np.random.randn(x) * np.pi

    phase = np.zeros_like(phase)

    # ndarray that stores these parameters
    params = np.zeros((x, 3))
    params[:, 0] = frequency
    params[:, 1] = amplitude
    params[:, 2] = phase

    input = np.zeros((x, IN))
    output = np.zeros((x, OUT))

    linespace = np.linspace(Delta[0], Delta[1], Delta[2])
    for i in range(x):
        data = np.sin(linespace * params[i, 0] + params[i, 2]) * params[i, 1]
        input[i, :] = data[:IN]
        output[i, :] = data[IN:]

    input = np.asarray(input,dtype=float)
    output = np.asarray(output,dtype=float)

    input_path, output_path = path(type, dir)

    np.save(input_path, input)
    np.save(output_path, output)
    print(type+" data saved to disk.")

    return input, output


def loadData(dir=dset_dir):
    """
    Returns data loaders from disk.
    """
    train_in_path, train_out_path = path('train', dir)
    val_in_path, val_out_path = path('val', dir)
    test_in_path, test_out_path = path('test', dir)

    train_in = np.load(train_in_path)
    train_out = np.load(train_out_path)
    val_in = np.load(val_in_path)
    val_out = np.load(val_out_path)
    test_in = np.load(test_in_path)
    test_out = np.load(test_out_path)

    dset_train = SineWaveDataset(train_in, train_out, train=True)
    dset_val = SineWaveDataset(val_in, val_out, train=True)
    dset_test = SineWaveDataset(test_in, test_out, train=False)

    loader_train = DataLoader(dset_train, batch_size=10)
    loader_val = DataLoader(dset_val, batch_size=10)
    loader_test = DataLoader(dset_test, batch_size=10)

    return loader_train, loader_val, loader_test


def path(type, dir):
    """
    Returns input/output dataset path.
    """
    in_path = os.path.join(dir,type+'_input.npy')
    out_path = os.path.join(dir,type+'_output.npy')
    return in_path, out_path


def visualize(input, gt, output=None, delta=Delta, index=0):
    """
    Plots a random sine wave from the given dataset.

    Inputs:
    - input: Input sine waves of shape (x, IN)
    - gt: ground truth output of shape (x, OUT)
    - output: Predicted output of shape (x, OUT), default None
    - delta: Linspace of the sine waves, number of separation = IN + OUT
    - index: index of the sine wave to show, or False to show all
    """
    x, IN = input.shape
    linspace = np.linspace(delta[0], delta[1], delta[2])
    if index == False:
        for i in range(x):
            plt.plot(linspace[:IN], input[i])
            plt.plot(linspace[IN:], gt[i])
            if output != None:
                plt.plot(linspace[IN:], output[i])
            plt.xlabel('Angle (rad)')
            plt.ylabel('A*sin(wt+phi)')
            plt.axis('tight')
            plt.show()
    elif type(index) == int:
        if index >= x:
            raise Exception("Index out of bound!")
        plt.plot(linspace[:IN], input[index])
        plt.plot(linspace[IN:], gt[index])
        if output != None:
            plt.plot(linspace[IN:], output[index])
        plt.xlabel('Angle (rad)')
        plt.ylabel('A*sin(wt+phi)')
        plt.axis('tight')
        plt.show()


def train(model, optimizer, epochs=10, filename=model_path):
    """
    Training loop.

    Inputs:
    - model: torch.nn.module
    - optimizer: torch.optim
    """
    model = model.to(device=device)
    criterion = nn.MSELoss()
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                # check_loss(loader_val, model)
                print()

    torch.save(model.state_dict(), filename)
    print("Model saved to disk.")


def check_loss(loader, model, visual=False):
    """
    Compute loss on validation/test set.

    Inputs:
    - loader: Dataloader containing data
    - model: torch.nn.module
    """
    if loader.dataset.train:
        print('Checking loss on Validation set')
    else:
        print('Checking loss on Test set')

    criterion = nn.MSELoss()
    loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            out = model(x)

            if visual:
                visualize(x, y, out, Delta, index=False)

            loss += criterion(out, y)
            count = t + 1

        print('Got average loss: %.4f' % (loss / count))


class SineWaveLSTM(nn.Module):

    def __init__(self, input_dim, embed_dim, fc2_dim, lstm_dim, hidden_dim, output_dim, sequence_num):
        """
        Inputs:
        - input_dim: input dimension
        - embed_dim: embedding dimension
        - hidden_dim: hidden state dimension
        - output_dim: output dimension
        - sequence_num: number of time steps of output
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.fc2_dim = fc2_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_num = sequence_num

        self.fc1 = nn.Linear(input_dim, embed_dim)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(embed_dim, fc2_dim)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(fc2_dim, lstm_dim)
        nn.init.kaiming_normal_(self.fc3.weight)

        self.lstm = nn.LSTM(lstm_dim, hidden_dim, batch_first=True,num_layers=2)
        # let weights of self.lstm be initialized by default

        self.fc4 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        """
        Inputs:
        - x: input of shape (batch_size, input_dim)

        Returns:
        - out: computed output (1D) on each time step (batch_size, sequence_num)
        """
        embed = self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))
        embed_stack = torch.stack([embed] * self.sequence_num, dim=1)
        lstm_out, _ = self.lstm(embed_stack)
        out = self.fc4(lstm_out)
        out = torch.reshape(out,(-1,self.sequence_num))
        return out


# Generate dataset
buildData(10, Delta, IN, OUT, 'train')
buildData(1, Delta, IN, OUT, 'val')
buildData(1, Delta, IN, OUT, 'test')

loader_train, loader_val, loader_test = loadData()

model = SineWaveLSTM(input_dim, embed_dim, fc2_dim, lstm_dim, hidden_dim, output_dim, sequence_num)

# Train/overfit a small dataset
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train(model=model, optimizer=optimizer, epochs=300)
check_loss(loader_train, model, visual=True)

# check performance on test
# model.load_state_dict(torch.load(model_path))
# check_loss(loader_train, model, visual=True)