print("Starting next imports")
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print("Finished next imports")


class Predictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_size, matrix):
        super(Predictor, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(matrix))
        self.lstm = torch.nn.LSTM(embedding_dim, 350, num_layers=1, bidirectional=True)

        self.linear = nn.Linear(700*2, 100)
        self.linear2 = nn.Linear(100, class_size)
    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1*2, 1, 350)).cuda(),
                torch.autograd.Variable(torch.zeros(1*2, 1, 350)).cuda())
    def forward(self, x1, x2):
        embeds1 = self.embeddings(x1)
        embeds2 = self.embeddings(x2)
        hidden1 = self.init_hidden()
        hidden2 = self.init_hidden()
        embeds1_out, embeds1_hidden = self.lstm(embeds1.view(x1.data.size()[0], 1, -1), hidden1)
        embeds1_avg = embeds1_out[-1].view(1, -1)
        embeds2_out, embeds2_hidden = self.lstm(embeds2.view(x2.data.size()[0], 1, -1), hidden2)
        embeds2_avg = embeds2_out[-1].view(1, -1)
        concat_vec = torch.cat((embeds1_avg, embeds2_avg), 1)
        #print(diff_vec)
        return (self.linear2(F.relu(self.linear(concat_vec))))