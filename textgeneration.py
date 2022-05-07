import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.distributions import Categorical
import numpy as np
from cifar10 import get_num_params


# Helper Functions for Training and Testing

def train(rnn_model, epoch, seq_len=200):   # seq_length is length of training data sequence
    rnn_model.train()
    loss_fn = nn.CrossEntropyLoss()         # loss function
    test_output_len = 200                   # total num of characters in output test sequence
    # random starting point in [0, seq_len-1] to partition data into chunks of length seq_len
    # This is Truncated Back-propagation Through Time
    data_ptr = np.random.randint(seq_len)
    running_loss, n = 0, 0

    if epoch % 10 == 0 or epoch == 1 or epoch == 2 or epoch == 3:
        print(f"\nStart of Epoch: {epoch}")

    while True:
        input_seq = data[data_ptr: data_ptr + seq_len]
        target_seq = data[data_ptr + 1: data_ptr + seq_len + 1]
        input_seq.to(device)
        target_seq.to(device)

        optimizer.zero_grad()
        output = rnn_model(input_seq)
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        data_ptr += seq_len     # update the data pointer
        if data_ptr + seq_len + 1 > data_size:
            break   # if at end of data then stop
        n += 1

    # print loss and a sample of generated text periodically
    if epoch % 10 == 0 or epoch == 1 or epoch == 2 or epoch == 3:
        # sample / generate a text sequence after every epoch
        rnn_model.eval()
        data_ptr = 0
        rand_index = np.random.randint(data_size - 1)   # random character from data to begin
        input_seq = data[rand_index: rand_index + 1]
        test_output = ""
        while True:
            output = rnn_model(input_seq)   # forward pass
            # construct categorical distribution and sample a character
            output = fun.softmax(torch.squeeze(output), dim=0)
            # output.to("cpu")
            dist = Categorical(output)
            index = dist.sample().item()
            test_output += ix_to_char[index]    # append the sampled character to test_output
            input_seq[0][0] = index             # next input is current output
            data_ptr += 1
            if data_ptr > test_output_len:
                break
        print("TRAIN Sample")
        print(test_output)
        print(f"End of Epoch: {epoch} \t Loss: {running_loss / n:.8f}")

    return running_loss / n


def test(rnn_model, output_len=1000):
    rnn_model.eval()
    data_ptr, hidden_state = 0, None    # initialize variables

    rand_index = np.random.randint(data_size - 11)  # randomly select an initial string from the data of 10 characters
    input_seq = data[rand_index: rand_index + 9]
    output = rnn_model(input_seq)                       # compute last hidden state of the sequence
    input_seq = data[rand_index + 9: rand_index + 10]   # next element is the input to rnn

    # generate remaining sequence
    # NOTE: We generate one character at a time
    test_output = ""
    while True:
        output = rnn_model(input_seq)       # forward pass
        output = fun.softmax(torch.squeeze(output), dim=0)  # construct categorical distribution and sample a character
        dist = Categorical(output)
        index = dist.sample().item()
        test_output += ix_to_char[index]    # append the sampled character to test_output
        input_seq[0][0] = index             # next input is current output
        data_ptr += 1
        if data_ptr > output_len:
            break
    print("\nTEST -------------------------------------------------")
    print(test_output)
    print("----------------------------------------")


class MyRNN(nn.Module):
    """RNN for text generation"""
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=3, do_dropout=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.do_dropout = do_dropout
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.hidden_state = None    # the hidden state of the RNN

    def forward(self, input_seq):
        x = nn.functional.one_hot(input_seq, self.input_size).float()
        if self.do_dropout:
            x = self.dropout(x)
        x, new_hidden_state = self.rnn(x, self.hidden_state)
        output = self.decoder(x)
        # save the hidden state for the next batch; detach removes extra datastructures for backprop etc.
        self.hidden_state = new_hidden_state.detach()
        return output

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except Exception as err:
            print("Error loading model from file", path)
            print(err)
            print("Initializing model weights to default")
            self.__init__(self.input_size, self.output_size, self.hidden_size, self.num_layers)


class MyLSTM(nn.Module):
    """LSTM for text generation"""
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=3, do_dropout=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.do_dropout = do_dropout
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.internal_state = None  # the internal state of the LTSM,
        # consists of the short term memory or hidden state and
        # the long term memory or cell state

    def forward(self, input_seq):
        x = nn.functional.one_hot(input_seq, self.input_size).float()
        if self.do_dropout:
            x = self.dropout(x)
        x, new_internal_state = self.lstm(x, self.internal_state)
        output = self.decoder(x)
        self.internal_state = (new_internal_state[0].detach(), new_internal_state[1].detach())
        return output

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except Exception as err:
            print("Error loading model from file", path)
            print(err)
            print("Initializing model weights to default")
            self.__init__(self.input_size, self.output_size, self.hidden_size, self.num_layers)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data_file = "./data/input.txt"                  # Load the data into memory and then open it
    data = open(data_file, 'r').read(20000)         # Read only ~20KB of data; full data takes long time in training
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)   # NOTE: vocab_size is a hyper-parameter of our models
    print(f"Data has {data_size} characters, {vocab_size} unique")

    char_to_ix = {ch: i for i, ch in enumerate(chars)}      # char to index map
    ix_to_char = {i: ch for i, ch in enumerate(chars)}      # index to char map

    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]            # convert data from chars to indices
    data = torch.tensor(data).to(device)    # data tensor on device
    data = torch.unsqueeze(data, dim=1)

    # Creating and Training an instance

    # First, plain RNNs
    model_save_file = "./model_data.pth"
    Model = MyRNN(vocab_size, vocab_size).to(device)
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.002)
    best_model_rnn = MyRNN(vocab_size, vocab_size).to(device)
    best_rnn_loss = 10000
    for epoch in range(0, 101):     # values from 1 to 100
        # model_rnn.load_model(model_save_file)
        epoch_loss = train(Model, epoch)
        if epoch_loss < best_rnn_loss:
            best_rnn_loss = epoch_loss
            best_model_rnn.load_state_dict(Model.state_dict())
        # if epoch % 10 == 0:
        #    model_rnn.save_model(model_save_file)

    # Some sample generated text
    print("best loss", best_rnn_loss)
    print("Model size", get_num_params(best_model_rnn))
    test(best_model_rnn)

    # Next, LSTMs
    seq_len = 100   # length of LSTM sequence
    model_save_file = "./model_data.pth"
    model_lstm = MyLSTM(vocab_size, vocab_size).to(device)
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.002)
    best_model_lstm = MyLSTM(vocab_size, vocab_size).to(device)
    best_lstm_loss = 10000
    for epoch in range(0, 101):     # values from 0 to 100
        # model_lstm.load_model(model_save_file)
        epoch_loss = train(model_lstm, epoch)
        if epoch_loss < best_lstm_loss:
            best_lstm_loss = epoch_loss
            best_model_lstm.load_state_dict(model_lstm.state_dict())
        # if epoch % 10 == 0:
        #    model_lstm.save_model(model_save_file)

    # Some sample generated text
    print("Model size", get_num_params(best_model_lstm), "best loss", best_lstm_loss)
    test(best_model_lstm)
    test(best_model_lstm)
    test(best_model_lstm)
    test(best_model_lstm)
