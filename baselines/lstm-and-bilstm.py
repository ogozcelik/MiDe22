# Author   : Oguzhan Ozcelik
# Date     : 19.08.2022
# Subject  : LSTM/BiLSTM model for text classification
# Framework: PyTorch

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.data import get_tokenizer
from sklearn.metrics import classification_report

print(f'Torch: {torch.__version__}')
print(f'Torchtext: {torchtext.__version__}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class LSTM(nn.Module):
    def __init__(self, save_path, dimension=50, embedding_size=125, bidirectional=False, target_size=3):
        super(LSTM, self).__init__()
        self.device = device
        self.save_path = os.path.realpath(save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.embedding = nn.Embedding(len(text_field.vocab), embedding_size)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.drop = nn.Dropout(p=0.5)
        if bidirectional:
            self.fc = nn.Linear(2 * dimension, target_size)
        else:
            self.fc = nn.Linear(dimension, target_size)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_backward = output[:, 0, self.dimension:]
        out_concat = torch.cat((out_forward, out_backward), 1)
        drop_out = self.drop(out_concat)
        fc_out = self.fc(drop_out)
        out_probs = torch.sigmoid(torch.squeeze(fc_out, 1))

        return out_probs


def save_checkpoint(model, optimizer, save_path):
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(state_dict, save_path)
    print(f'Model and optimizer saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model and optimizer loaded from ==> {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])


def save_metrics(train_losses, global_steps_list, save_path):
    state_dict = {'train_losses': train_losses,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path, device):
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_losses'], state_dict['global_steps_list']


def train(model,
          optimizer,
          train_loader,
          eval_every,
          criterion=nn.CrossEntropyLoss(),
          num_epochs=20):

    running_loss = 0.0
    running_acc = 0.0
    global_step = 0
    train_losses, global_step_list, train_acc = [], [], []
    print("Training...")
    model.train()
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit='batch') as train_iter_:
            for ((text, text_len), label), _ in train_iter_:
                train_iter_.set_description(f"Epoch {epoch+1}/{num_epochs}")

                label = label.to(device)
                text = text.to(device)
                text_len = text_len.to(device)

                output = model(text, text_len)
                loss = criterion(output, label)

                preds = torch.argmax(output, dim=1)
                correct = (preds == label).sum().item()
                acc = correct / len(label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_iter_.set_postfix(loss=loss.item(), accuracy=100 * acc)

                running_loss += loss.item()
                running_acc += acc
                global_step += 1
                if global_step % eval_every == 0:
                    average_train_loss = running_loss / eval_every
                    average_train_acc = running_acc / eval_every
                    train_losses.append(average_train_loss)
                    train_acc.append(average_train_acc)
                    global_step_list.append(global_step)
                    running_loss = 0.0
                    running_acc = 0.0
                    model.train()

    print('Training done!')
    return train_losses, train_acc, global_step_list


def evaluate(model, test_loader):
    y_pred, y_true = [], []
    print('Evaluation...')
    model.eval()
    with torch.no_grad():
        for ((text, text_len), label), _ in test_loader:
            label = label.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            y_pred.extend(torch.argmax(output, dim=1).tolist())
            y_true.extend(label.tolist())
    print('Evaluation done!')
    return y_true, y_pred


for lang in ['EN', 'TR']:
    print(f"Language: {'English' if lang=='EN' else 'Turkish'}")

    path = 'results/BiLSTM/' + lang
    if not os.path.exists(path):
        os.makedirs(path)

    for fold in range(5):
        print(f"Fold: {fold}")
        tokenizer = get_tokenizer(tokenizer=None)  # just splits the sentence
        label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
        text_field = Field(tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True)
        fields = [('text', text_field), ('label', label_field)]

        train_data = TabularDataset(path=os.path.join('./dataset', lang, 'folds', lang+'_train_'+str(fold)+'.tsv'),
                                    format='tsv',
                                    fields=fields,
                                    skip_header=True)

        test_data = TabularDataset(path=os.path.join('./dataset', lang, 'folds', lang+'_test_'+str(fold)+'.tsv'),
                                   format='tsv',
                                   fields=fields,
                                   skip_header=True)

        train_iter = BucketIterator(dataset=train_data, batch_size=16, sort_key=lambda x: len(x.text),
                                    device=device, sort=True, sort_within_batch=True)
        test_iter = BucketIterator(dataset=test_data, batch_size=16, sort_key=lambda x: len(x.text),
                                   device=device, sort=True, sort_within_batch=True)

        text_field.build_vocab(train_data, min_freq=3)

        model = LSTM(save_path='').to(device)  # set bidirectional==True for BiLSTM
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses, train_acc, global_step_list = train(model=model, optimizer=optimizer, train_loader=train_iter,
                                                          eval_every=len(train_iter) // 2)
        y_true, y_pred = evaluate(model=model, test_loader=test_iter)

        report = classification_report(y_true, y_pred, digits=4)
        print(report)

        with open(os.path.join(path, 'classification_report_'+str(fold)), 'w') as file:
            file.write(report)