import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torchnlp.word_to_vector import GloVe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCHNLP_CACHEDIR = "D:/PyEnvs/pytorch-nlp_data"
pretrained_wv = GloVe(cache=TORCHNLP_CACHEDIR)


class EDCDataset(Dataset):
    def __init__(self, inputs, targets, istrain):
        self.inputs = inputs
        self.targets = targets
        self.istrain = istrain

    def __getitem__(self, index):
        input_ = []
        for sent in self.inputs[index]:
            input_.append(torch.stack([pretrained_wv[word] for word in sent]).to(DEVICE))
        input_ = pack_sequence(input_, enforce_sorted=False)
        target = torch.LongTensor(self.targets[index]).to(DEVICE)

        return (input_, target, target.size(0))

    def __len__(self):
        return len(self.inputs)


def collate_fn(data):
    inputs, targets, lengths = map(list, zip(*data))
    targets = pad_sequence(targets, batch_first=True)
    masks = torch.zeros(targets.size(), dtype=torch.bool).to(DEVICE)
    for i, length in enumerate(lengths):
        masks[i][0:length] = 1

    return (inputs, targets, masks)


def build_dataset(data_path, label_path, istrain=True):
    """
    Returns:
        dataset
    """

    inputs = []
    targets = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            _sents = line.strip("\n").strip(" __eou__").split(" __eou__")
            sents = []
            for sent in _sents:
                words = sent.split(" ")
                sents.append(words)
            inputs.append(sents)

    with open(label_path, "r", encoding="utf8") as f:
        for line in f:
            labels = [int(label) for label in line.strip("\n").strip(" ").split(" ")]
            targets.append(labels)

    dataset = EDCDataset(inputs, targets, istrain)
    return dataset


if __name__ == "__main__":
    build_dataset("./data/Emotion Detection in Conversations/test/dialogues_test.txt",
                  "./data/Emotion Detection in Conversations/test/dialogues_emotion_test.txt")
