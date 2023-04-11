import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    def __init__(self, device, n_features, n_hidden, n_outputs, sequence_len, n_lstm=1, n_deep=2):
        super().__init__()

        self.device = device
        
        self.n_lstm = n_lstm
        self.n_hid = n_hidden

        self.lstm = nn.LSTM(n_features, n_hidden, num_layers=n_lstm, batch_first=True)

        self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)



    def forward(self, x):

        # Initialize hidden state and cell state
        hidden_state = torch.zeros(self.n_lstm, x.shape[0], self.n_hid)
        cell_state = torch.zeros(self.n_lstm, x.shape[0], self.n_hid)
        hidden_state, cell_state = hidden_state.to(self.device), cell_state.to(self.device)

        self.hidden = (hidden_state, cell_state)

        # Forward pass
        x, h = self.lstm(x, self.hidden)
        x = x.contiguous().view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)