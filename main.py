import argparse
import random

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from torch.utils.data import DataLoader

import models
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro")
    r = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred)

    return (acc, p, r, f1, report)


def train(model, train_data_loader, criterion, optimizer):
    model.train()
    loss_list = []
    pred_list = []
    true_list = []
    for inputs, targets, masks in train_data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        class_num = outputs.size(-1)
        outputs = torch.masked_select(outputs, masks.unsqueeze(-1)).view(-1, class_num)
        targets = torch.masked_select(targets, masks)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
        true_list.append(targets.cpu().numpy())
        
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(true_list)

    loss = np.mean(loss_list)
    result = eval_metrics(y_true, y_pred)

    return (loss, *result)


def evaluate(model, eval_data_loader, criterion):
    model.eval()
    loss_list = []
    pred_list = []
    true_list = []
    with torch.no_grad():
        for inputs, targets, masks in eval_data_loader:
            outputs = model(inputs)

            class_num = outputs.size(-1)
            outputs = torch.masked_select(outputs, masks.unsqueeze(-1)).view(-1, class_num)
            targets = torch.masked_select(targets, masks)
            loss = criterion(outputs, targets)

            loss_list.append(loss.item())
            pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            true_list.append(targets.cpu().numpy())

    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(true_list)

    loss = np.mean(loss_list)
    result = eval_metrics(y_true, y_pred)

    return (loss, *result)


if __name__ == "__main__":
    # path config
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default="./data/Emotion Detection in Conversations/train/dialogues_train.txt")
    parser.add_argument("--train_label_path", type=str,
                        default="./data/Emotion Detection in Conversations/train/dialogues_emotion_train.txt")

    parser.add_argument("--dev_data_path", type=str,
                        default="./data/Emotion Detection in Conversations/validation/dialogues_validation.txt")
    parser.add_argument("--dev_label_path", type=str,
                        default="./data/Emotion Detection in Conversations/validation/dialogues_emotion_validation.txt")

    parser.add_argument("--test_data_path", type=str,
                        default="./data/Emotion Detection in Conversations/test/dialogues_test.txt")
    parser.add_argument("--test_label_path", type=str,
                        default="./data/Emotion Detection in Conversations/test/dialogues_emotion_test.txt")
    parser.add_argument("--model_save_path", type=str, default="./model/model.pt")
    parser.add_argument("--istrain", action="store_true")

    # model config
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--lstm_size", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)

    # seed
    parser.add_argument("--seed", type=int, default=2)
    opts = parser.parse_args()
    print(opts)

    # fix random seeds
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    # build dataset
    if opts.istrain:
        train_dataset = utils.build_dataset(opts.train_data_path, opts.train_label_path)
        dev_dataset = utils.build_dataset(opts.dev_data_path, opts.dev_label_path)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, collate_fn=utils.collate_fn, shuffle=True)
        dev_data_loader = DataLoader(dataset=dev_dataset, batch_size=opts.batch_size, collate_fn=utils.collate_fn, shuffle=False)
    test_dataset = utils.build_dataset(opts.test_data_path, opts.test_label_path)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, collate_fn=utils.collate_fn, shuffle=False)

    # build model
    model = models.Model(opts.embedding_size, opts.lstm_size, opts.hidden_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), opts.learning_rate)

    if opts.istrain:
        best_f1 = 0
        for i in range(opts.epochs):
            print("Epoch: {} ################################".format(i))
            train_loss, train_acc, train_p, train_r, train_f1, _ = train(model, train_data_loader, criterion, optimizer)
            dev_loss, dev_acc, dev_p, dev_r, dev_f1, _ = evaluate(model, dev_data_loader, criterion)
            print("Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(train_loss, train_acc, train_f1, train_p, train_r))
            print("Dev   Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(dev_loss, dev_acc, dev_f1, dev_p, dev_r))
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                torch.save(model.state_dict(), opts.model_save_path)
            print("###########################################")
    model.load_state_dict(torch.load(opts.model_save_path))
    test_loss, test_acc, test_p, test_r, test_f1, result = evaluate(model, test_data_loader, criterion)
    print("Test   Loss: {:.4f} Acc: {:.4f} F1: {:.4f}({:.4f}/{:.4f})".format(test_loss, test_acc, test_f1, test_p, test_r))
    print(result)
