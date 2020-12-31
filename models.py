import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pack_sequence,
                                pad_packed_sequence, pad_sequence)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtraction(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.bilstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        """
        inputs: B*(N, max(L))
        outputs: (B, N, U), lengths
        """
        outputs = []
        for input_ in inputs:
            output, _ = self.bilstm(input_)  # (N, max(L), D) -> (N, max(L), 2*H)
            output, lengths = pad_packed_sequence(output, batch_first=True)  # (N, max(L), 2*H)
            output = output.view(output.size(0), output.size(1), 2, -1)
            output = torch.stack([torch.cat([output[i, length-1, 0], output[i, 0, 1]])
                                  for i, length in enumerate(lengths)])  # (N, max(L), 2*H) -> (N, 2*H)
            outputs.append(output)

        outputs = pad_sequence(outputs, batch_first=True)  # (B, max(N), U)
        return outputs


class EmotionDetection(nn.Module):
    def __init__(self, utterance_size, hidden_size):
        super().__init__()
        self.global_gru = nn.GRUCell(utterance_size, utterance_size)
        self.global_W = nn.Parameter(torch.randn(utterance_size, utterance_size))
        self.party_gru = nn.GRUCell(utterance_size, utterance_size)
        self.emotion_gru = nn.GRUCell(utterance_size, hidden_size)
        self.utterance_size = utterance_size
        self.hidden_size = hidden_size

    def forward(self, inputs):  # (B, N, U)
        inputs = inputs.transpose(0, 1)  # (N, B, U)
        global_s = [torch.zeros(inputs.size(1), self.utterance_size).to(DEVICE)]
        party_s = [torch.zeros(inputs.size(1), self.utterance_size).to(DEVICE),
                   torch.zeros(inputs.size(1), self.utterance_size).to(DEVICE)]
        emotion_s = torch.zeros(inputs.size(1), self.hidden_size).to(DEVICE)

        outputs = []
        for i in range(inputs.size(0)):
            ut = inputs[i]  # (B, U)
            cur_party = i % 2

            global_s.append(self.global_gru(ut+party_s[cur_party], global_s[i]))

            # 1+n means additional init state g(0)
            global_s_history = torch.stack(global_s[0:i+1])  # (1+n, B, U)
            attn_score = (ut.unsqueeze(1) @ self.global_W @ global_s_history.unsqueeze(-1)).squeeze(-1)
            alpha = F.softmax(attn_score, dim=0)  # (1+n, B, 1)
            ct = torch.sum(global_s_history*alpha, dim=0)  # (B, U)

            party_s[cur_party] = self.party_gru(ut+ct, party_s[cur_party])
            emotion_s = self.emotion_gru(party_s[cur_party], emotion_s)
            outputs.append(emotion_s)

        outputs = torch.stack(outputs).transpose(0, 1)  # (B, N, H)
        return outputs


class Model(nn.Module):
    def __init__(self, embedding_size, lstm_size, hidden_size, output_size=7):
        super().__init__()
        self.feature_extraction = FeatureExtraction(embedding_size, lstm_size)
        self.emotion_detection = EmotionDetection(2*lstm_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        outputs = self.feature_extraction(inputs)  # (B, pack(N, L, D)) -> (B, N, U)
        outputs = self.emotion_detection(outputs)  # (B, N, U) -> (B, N, H)
        outputs = self.linear(outputs)  # (B, N, H) -> (B, N, C)
        # use CrossEntropyLoss
        return outputs
