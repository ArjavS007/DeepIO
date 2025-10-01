import torch.nn as nn

# class LSTMModelV3(nn.Module):
#     def __init__(self, in_dim=10, hidden_size=250, num_layers=1, output_size=3, dropout_prob=0.1):
#         super(LSTMModelV3, self).__init__()
#         # Initialize hidden and cell states
#         # self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
#         # self.c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

#         self.name='LSTMModelV3'
#         self.lstm_1 = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True) #, return_sequences=True)
#         self.dropout_1 = nn.Dropout(dropout_prob)
#         self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.dropout_2 = nn.Dropout(dropout_prob)
#         self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.dropout_3 = nn.Dropout(dropout_prob)
#         self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.dropout_4 = nn.Dropout(dropout_prob)
#         self.lstm_5 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.dropout_5 = nn.Dropout(dropout_prob)
#         self.lstm_6 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x, _ = self.lstm_1(x)
#         x = self.dropout_1(x)
#         x, _ = self.lstm_2(x)
#         x = self.dropout_2(x)
#         x, _ = self.lstm_3(x)
#         x = self.dropout_3(x)
#         x, _ = self.lstm_4(x)
#         output = self.fc(x[:, -1, :])
#         return output


class LSTMModelV3(nn.Module):
    def __init__(
        self, in_dim=10, hidden_size=500, num_layers=1, output_size=3, dropout_prob=0.1
    ):
        super(LSTMModelV3, self).__init__()
        # Initialize hidden and cell states
        # self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        # self.c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

        self.name = "LSTMModelV3"
        self.lstm_1 = nn.LSTM(
            in_dim, hidden_size, num_layers, batch_first=True
        )  # , return_sequences=True)
        # self.dropout_1 = nn.Dropout(dropout_prob)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_2 = nn.Dropout(dropout_prob)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_3 = nn.Dropout(dropout_prob)
        self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_4 = nn.Dropout(dropout_prob)
        # self.lstm_5 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_5 = nn.Dropout(dropout_prob)
        # self.lstm_6 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        # x = self.dropout_1(x)
        x, _ = self.lstm_2(x)
        # x = self.dropout_2(x)
        x, _ = self.lstm_3(x)
        # x = self.dropout_3(x)
        x, _ = self.lstm_4(x)
        output = self.fc(x[:, -1, :])
        return output


class LSTMModel3L(nn.Module):
    def __init__(
        self, in_dim=10, hidden_size=500, num_layers=1, output_size=3, dropout_prob=0.1
    ):
        super(LSTMModel3L, self).__init__()
        # Initialize hidden and cell states
        # self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        # self.c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

        self.name = "LSTMModel3L"
        self.lstm_1 = nn.LSTM(
            in_dim, hidden_size, num_layers, batch_first=True
        )  # , return_sequences=True)
        # self.dropout_1 = nn.Dropout(dropout_prob)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_2 = nn.Dropout(dropout_prob)
        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_3 = nn.Dropout(dropout_prob)
        # self.lstm_4 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_4 = nn.Dropout(dropout_prob)
        # self.lstm_5 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.dropout_5 = nn.Dropout(dropout_prob)
        # self.lstm_6 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        # x = self.dropout_1(x)
        x, _ = self.lstm_2(x)
        # x = self.dropout_2(x)
        x, _ = self.lstm_3(x)
        # x = self.dropout_3(x)
        # x, _ = self.lstm_4(x)
        output = self.fc(x[:, -1, :])
        return output
