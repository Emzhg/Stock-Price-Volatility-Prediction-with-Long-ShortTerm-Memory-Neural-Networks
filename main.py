import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

## cours du 25/03



class CustomLSTM(nn.Module):
    def __init__(
            self,
            input_size : int,
            hidden_size : int
    ):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.input_size))

        self.w_f = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.input_size))


        self.w_c = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(self.input_size))


        self.w_o = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.input_size))

        self.init()

    def init(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(
            self,
            X : torch.Tensor
    ):
        bs, seq_size, _ = X.size()
        hidden_seq = []

        h_t, c_t = (
            torch.zeros(bs, self.hidden_size),
            torch.zeros(bs, self.hidden_size)
        )
        for t in range(seq_size):
            x_t = X[:,t,:]
            i_t = torch.sigmoid(x_t @ self.w_i + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.w_f + h_t @ self.u_f + self.b_f)
            g_t = torch.tanh(x_t @ self.w_c + h_t @ self.u_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.w_o + h_t @ self.u_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim = 0)
        hidden_seq = hidden_seq.transpose(0,1).contigous()
        return hidden_seq, (h_t, c_t)

lstm = CustomLSTM(input_size=20, hidden_size=2)
X = torch.from_numpy(np.random.normal(size=(1000,20)))
#X = Variable(torch.reshape(X,(X.shape[0],1, X.shape[1])).float())
#lstm.forward(X)
# ces deux lignes ne fonctionnent pas à voir pourquoi

class LSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM,self).__init__()
        #Hidden dimensions
        self.hidden_dim = hidden_dim

        #Number of hidden layers
        self.num_layers = num_layers

        # batch_first = True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim-
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # Initialize hidden state with zeros

        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_dim).requires_grad_()

        # Initalize cell state
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation trough time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(X, (h0.detach(),c0.detach()))

        # Index hidden state ou last time step
        # out.size() --> 100,32,100
        # out[:, -1, : ] --> 100,100 --> juste want last time step hidden states !
        out = self.fc(out[:,-1,:])
        # out.size() --> 100,10
        return out

#target variable en première colonne / variable cible en première colonne // stock
def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index : index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    # à normaliser
    scaler = StandardScaler()

    #train_set_size_norm = scaler.fit(train_set_size)
    x_train = data[:train_set_size]
    y_train = data[:train_set_size]



    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1,1)

    x_test = data[train_set_size:,:-1,:]
    y_test = data[train_set_size:,-1,0].reshape(-1,1)

    return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
    data = pd.read_csv("/home/API/DL")
    data = data[["Close", "Volume"]]
    x_train, y_train, x_test, y_test = load_data(data, 5)
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    #model = LSTM(3,4,2,1) # 3 si 3 inputs, ici juste close et volume
    model = LSTM(2,4,2,1)
    optimiser = torch.optim.Adam(model.parameters(),lr = 0.01)
    #scheduler torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')
    print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    hist = np.zeros(1000)
    loss_fn = torch.nn.MSELoss()
    for t in range(1000):
        model.train()
        #Initialize hidden state
        #Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden()

        #Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        #Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t!=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        #Backward pass
        loss.backward()

        #Update parameters
        optimiser.step()

        # scheduler.step(loss.item())

    plt.plot(hist, label = "Training loss")
    plt.title('MSE loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend
    plt.grid()
    plt.show()


    end = True


