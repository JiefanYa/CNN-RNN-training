import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

device = torch.device('cpu')
dtype = torch.float32

# wandb.init(entity="wandb", project="lstm-sine")
# TODO: finish wandb setup


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


def buildData(x, delta, in_num, out_num, type, dir):
    """
    Generates random sine waves.

    Input:
    - x: Number of sine waves produced
    - delta: Linspace of the sine waves, number of separation = in_num, out_num
    - in_num: Number of input numbers in the interval defined by delta
    - out_num: Number of output numbers in the interval defined by delta
    - type: data type (train/val/test)
    - dir: directory to save data

    Returns a tuple of:
    - input: Input sine waves of shape (x, IN)
    - output: Output sine waves of shape (x, OUT)
    """
    # frequency w
    frequency = np.random.randn(x) + 1
    frequency[frequency == 0] = 1

    # frequency = np.ones_like(frequency)

    # amplitude A
    amplitude = (np.random.randn(x) + 1) * 2
    amplitude[amplitude == 0] = 1
    # phase Phi
    phase = np.random.randn(x) * np.pi

    # phase = np.zeros_like(phase)

    # ndarray that stores these parameters
    params = np.zeros((x, 3))
    params[:, 0] = frequency
    params[:, 1] = amplitude
    params[:, 2] = phase

    input = np.zeros((x, in_num))
    output = np.zeros((x, out_num))

    linespace = np.linspace(delta[0], delta[1], delta[2])
    for i in range(x):
        data = np.sin(linespace * params[i, 0] + params[i, 2]) * params[i, 1]
        input[i, :] = data[:in_num]
        output[i, :] = data[in_num:]

    input = np.asarray(input,dtype=float)
    output = np.asarray(output,dtype=float)

    input_path, output_path = path(type, dir)

    np.save(input_path, input)
    np.save(output_path, output)
    print(type+" data saved to disk.")

    return input, output


def loadData(dir, batch_size=64):
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

    loader_train = DataLoader(dset_train, batch_size=batch_size)
    loader_val = DataLoader(dset_val, batch_size=batch_size)
    loader_test = DataLoader(dset_test, batch_size=batch_size)

    return loader_train, loader_val, loader_test


def path(type, dir):
    """
    Returns input/output dataset path.
    """
    in_path = os.path.join(dir,type+'_input.npy')
    out_path = os.path.join(dir,type+'_output.npy')
    return in_path, out_path


def visualize(input, gt, delta=None, output=None, index=None):
    """
    Plots a random sine wave from the given dataset.

    Inputs:
    - input: Input sine waves of shape (x, IN)
    - gt: ground truth output of shape (x, OUT)
    - output: Predicted output of shape (x, OUT), default None
    - delta: Linspace of the sine waves, number of separation = IN + OUT
    - index: array of indices of sine waves to show, or False to show all,
            or a number of random sine waves to show
    """
    if index is None:
        index = [0]
    if delta is None:
        delta = [0, 2 * np.pi, 50]
    x, IN = input.shape
    linspace = np.linspace(delta[0], delta[1], delta[2])
    indices = []
    if index == False:
        indices = list(range(x))
    elif type(index) == list:
        indices = index
    elif type(index) == int:
        if index > x:
            raise Exception('index out of bound')
        indices = np.random.randint(0,x,size=index)
    for i in indices:
        plt.plot(linspace[:IN], input[i])
        plt.plot(linspace[IN:], gt[i])
        if output != None:
            plt.plot(linspace[IN:], output[i])
        plt.xlabel('Angle (rad)')
        plt.ylabel('A*sin(wt+phi)')
        plt.axis('tight')
        plt.show()


def train(model, optimizer, loader_train, loader_val, file, epochs=10, print_every=100):
    """
    Training loop.

    Inputs:
    - model: torch.nn.module
    - optimizer: torch.optim
    - loader_train: training dataloader
    - loader_val: validation dataloader
    - file: path to save trained model
    - epochs: number of epochs to train
    - print_every: accuracy logging frequency
    """
    print("Training starts.")
    print()
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

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                check_loss(loader_val, model)
                print()

    torch.save(model.state_dict(), file)
    print("Training complete. Model saved to disk.")
    print()


def check_loss(loader, model, delta=None):
    """
    Compute loss on validation/test set.

    Inputs:
    - loader: Dataloader containing data
    - model: torch.nn.module
    - delta: linspace to aid visualization, None to stop drawing charts
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

            loss += criterion(out, y)
            count = t + 1

        print('Got average loss: %.4f' % (loss / count))

    # get a random batch from loader, assume batch_size > 10
    if delta != None:
        print("Drawing charts.")
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=dtype)

                out = model(x)
                visualize(x, y, delta, out, index=10)
                break
        print("Charts complete.")


class SubMLP(nn.Module):

    def __init__(self, lstm_params):
        super().__init__()

        self.num_layers = lstm_params['num_mst_layers']
        self.layers = {}
        for i in range(self.num_layers-1):
            if i==0:
                input_dim = lstm_params['delta'][3]
                output_dim = lstm_params['l' + str(i+1) + '_out_dim']
            else:
                input_dim = lstm_params['l' + str(i) + '_out_dim']
                output_dim = lstm_params['l' + str(i+1) + '_out_dim']
            self.layers['fc'+str(i+1)] = nn.Linear(input_dim, output_dim)
            nn.init.kaiming_normal_(self.layers['fc'+str(i+1)].weight)
            self.layers['relu'+str(i+1)] = nn.ReLU()

        input_dim = lstm_params['l'+str(self.num_layers-1)+'_out_dim']
        output_dim = lstm_params['lstm_dim']
        self.layers['fc'+str(self.num_layers)] = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_normal_(self.layers['fc'+str(self.num_layers)].weight)

    def forward(self, x):
        retval = x
        for i in range(self.num_layers-1):
            retval = self.layers['fc'+str(i+1)](retval)
            retval = self.layers['relu'+str(i+1)](retval)
        retval  = self.layers['fc'+str(self.num_layers)](retval)
        return retval


class SineWaveLSTM(nn.Module):

    def __init__(self, lstm_params):
        """
        Inputs:
        - input_dim: input dimension
        - embed_dim: embedding dimension
        - hidden_dim: hidden state dimension
        - output_dim: output dimension
        - sequence_num: number of time steps of output
        """
        super().__init__()

        self.num_mst_layers = lstm_params['num_mst_layers']
        self.num_lstm_layers = lstm_params['num_lstm_layers']
        self.input_dim = lstm_params['delta'][3]
        self.sequence_num = lstm_params['delta'][4]
        self.lstm_dim = lstm_params['lstm_dim']
        self.hidden_dim = lstm_params['hidden_dim']

        self.mlp = SubMLP(lstm_params)
        self.lstm = nn.LSTM(self.lstm_dim, self.hidden_dim, batch_first=True, num_layers=self.num_lstm_layers)

        self.fc = nn.Linear(self.hidden_dim, 1)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        """
        Inputs:
        - x: input of shape (batch_size, input_dim)

        Returns:
        - out: computed output (1D) on each time step (batch_size, sequence_num)
        """
        embed = self.mlp(x)
        embed_stack = torch.stack([embed] * self.sequence_num, dim=1)
        lstm_out, _ = self.lstm(embed_stack)
        out = self.fc(lstm_out)
        out = torch.reshape(out,(-1,self.sequence_num))
        return out


def runLSTM(
    data_params,
    lstm_params,
    dir=os.path.dirname(__file__),
    new_data=True,
    batch_size=64,
    load_model=False,
    training=True,
    overfit=False,
    learning_rate=4e-3,
    epochs=15,
    print_every=100
):

    dataset_dir = os.path.join(dir, './data')
    model_path = os.path.join(dir, 'model.pt')

    train_num = data_params['train_num']
    val_num = data_params['val_num']
    test_num = data_params['test_num']
    delta = data_params['delta'][:3]
    in_num = data_params['delta'][3]
    out_num = data_params['delta'][4]

    if new_data:
        # Generate dataset
        buildData(train_num, delta, in_num, out_num, 'train', dataset_dir)
        buildData(val_num, delta, in_num, out_num, 'val', dataset_dir)
        buildData(test_num, delta, in_num, out_num, 'test', dataset_dir)

    # load data
    loader_train, loader_val, loader_test = loadData(dataset_dir, batch_size=batch_size)

    # initialize model
    model = SineWaveLSTM(lstm_params)

    if load_model:
        model.load_state_dict(torch.load(model_path))

    if training:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train(model, optimizer, loader_train, loader_val, model_path, epochs=epochs, print_every=print_every)

    if overfit:
        # check on train data
        check_loss(loader_train, model, delta=delta)
    else:
        # check on test data
        check_loss(loader_test, model, delta=delta)


if __name__ == "__main__":

    data_params = {}
    data_params['train_num'] = 5000
    data_params['val_num'] = 500
    data_params['test_num'] = 500
    data_params['delta'] = [0,2*np.pi,50,20,30]

    lstm_params = {}
    lstm_params['delta'] = [0,2*np.pi,50,20,30]
    lstm_params['num_mst_layers'] = 3
    lstm_params['num_lstm_layers'] = 2
    lstm_params['l1_out_dim'] = 256
    lstm_params['l2_out_dim'] = 128
    lstm_params['lstm_dim'] = 16
    lstm_params['hidden_dim'] = 512

    runLSTM(data_params=data_params,
            lstm_params=lstm_params,
            new_data=False,
            load_model=True,
            training=False,
            overfit=False,
            epochs=15)

