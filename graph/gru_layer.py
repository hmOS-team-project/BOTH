import torch
import torch.nn as nn

# Define GRU as intra-option policies
class GRU_CellLayer(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(GRU_CellLayer, self).__init__()
        self.device = device
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size).to(self.device)
        
    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        hidden_state = torch.zeros(batch_size, self.n_hidden, requires_grad=True).to(self.device)
        return hidden_state   
    
    def forward(self, x, hidden):
        # 前向传播逻辑
        x = x.to(self.device)
        hidden = hidden.to(self.device)
        
        hx = self.gru_cell(x, hidden)
        
        return hx, hx  # 返回当前隐藏状态和下一时间步的隐藏状态



if __name__ == '__main__':
    # n_features = 2 # this is number of parallel inputs
    # n_timesteps = 1 # this is number of timesteps

    # # convert dataset into input/output
    # X, y = split_sequences(dataset, n_timesteps)
    # print(X.shape, y.shape)

    # # create NN
    # lstm_net = LSTM_Layer(n_features,n_timesteps)
    # criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
    # optimizer = torch.optim.Adam(lstm_net.parameters(), lr=1e-1)

    # train_episodes = 500
    # batch_size = 16
    rnn = GRU_CellLayer((2,5), 20) # (input_size, hidden_size)
    input = torch.randn(2, 3, 2, 5) # (time_steps, batch, input_size[0], input_size[1])
    hx, cx = rnn.init_hidden(3)
    output = []
    for i in range(input.size()[0]):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)
    output = torch.stack(output, dim=0)
    print(output.shape)
    pass