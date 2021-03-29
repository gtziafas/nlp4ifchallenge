from ..types import *
from ..preprocessing import read_labeled
from .utils.embeddings import make_word_embedder, WordEmbedder
from .training import train_epoch, eval_epoch

import torch
import os
from torch import cat, stack
from torch.nn.utils.rnn import pad_sequence
from torch.nn import GRU, Dropout
from adabelief_pytorch import AdaBelief

SAVE_PATH = '../checkpoints'


class MultiLabelRNN(Module):
    def __init__(self, 
                num_classes: int, 
                inp_dim: int, 
                hidden_dim: int, 
                num_layers: int,
                dropout: float):
        super().__init__()
        self.rnn = GRU(input_size=inp_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.cls = Linear(in_features=2 * hidden_dim, out_features=num_classes) 
        self.dropout = Dropout(p=dropout)

    def forward (self, x: Tensor) -> Tensor:
        dh = self.rnn.hidden_size
        h, _ = self.rnn(x)
        context = cat((h[:, -1, :dh], h[:, 0, dh:]), dim=-1)
        context = self.dropout(context)
        out = self.cls(context)
        return out 


class TrainedMultiLabelRNN(Module, Model):
    def __init__(self,  
                core: MultiLabelRNN,
                word_embedder: WordEmbedder,
                device: str):
        super().__init__()
        self.we = word_embedder
        self.core = core
        self.device = device

    def embedd(self, tweets: List[Tweet]) -> Tensor:
        word_embedds = [tensor(self.we(tweet.text), dtype=torch.float, device=self.device) for tweet in tweets]
        return pad_sequence(word_embedds, batch_first=True)
        
    def predict(self, tweets: List[Tweet]) -> List[str]:
        inputs = self.embedd(tweets)
        preds = self.core.forward(inputs).sigmoid().round().long().cpu().tolist()
        return list(map(preds_to_str, preds))

    def predict_scores(self, tweets: List[Tweet]) -> array:
        inputs = self.embedd(tweets)
        return self.core.forward(inputs).sigmoid().cpu().tolist()


def default_rnn():
    return MultiLabelRNN(7, 300, 150, 1, 0.25)


def labels_to_vec(labels: List[Label]) -> array:
    return array([1 if l == True else 0 for l in labels])


def collator(batch: List[Tuple[array, array]]) -> Tuple[Tensor, Tensor]:
    xs, ys = zip(*batch)
    xs = pad_sequence([tensor(x, dtype=torch.float) for x in xs], batch_first=True)
    ys = stack([tensor(y, dtype=torch.long) for y in ys], dim=0)
    return xs, ys 


def train_rnn(train_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_train.tsv', 
        dev_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_dev_input.tsv', 
        test_path: str = '',
        batch_size: int = 16, 
        num_epochs: int = 50,
        val_feq: int = 5,
        embeddings: str = 'glove_lg',
        early_stop_patience: int = 6,
        device: str = 'cuda'):

    # random
    torch.manual_seed(0)

    # data
    train_data = read_labeled(train_path)
    dev_data = read_labeled(dev_path)

    # representation
    we = make_word_embedder(embeddings)
    train_data = [(we(tweet.text), labels_to_vec(tweet.labels)) for tweet in train_data]
    dev_data = [(we(tweet.text), labels_to_vec(tweet.labels)) for tweet in dev_data]
    train_dl = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=collator)
    dev_dl = DataLoader(dev_data, shuffle=True, batch_size=batch_size, collate_fn=collator)

    # model, optim, loss 
    model = default_rnn().to(device)
    model_name = '_'.join([embeddings, 'rnn', '.p'])
    optim = AdaBelief(model.parameters(), lr=1e-03, weight_decay=1e-02)
    loss_fn = BCEWithLogitsLoss(reduction='mean').to(device)

    train_log, dev_log = [], []
    best, patience = {'mean_f1': 0.}, early_stop_patience
    for epoch in range(num_epochs):
        train_log.append(train_epoch(model, train_dl, optim, loss_fn, device))
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'{train_log[-1]}')
        if (epoch+1) % val_feq == 0 or not epoch:
            print()
            print('----' * 50)
            dev_log.append(eval_epoch(model, dev_dl, loss_fn, device))
            print(f'EVALUATION') 
            print(f'{dev_log[-1]}')    
            print('----' * 50)
            print()

            # model selection
            # model selection
            if dev_log[-1]['mean_f1'] <= best['mean_f1']:
                patience -= 1
                if not patience:
                    print('\nEarly stopping...')
                    break
            else:
                torch.save(model.state_dict, os.path.join(SAVE_PATH, model_name))
                best = dev_log[-1]
                patience = early_stop_patience

    # save all stuff
    state_dict = torch.load(os.path.join(SAVE_PATH, model_name))
    all_stuff = {'model_state_dict': state_dict, 
                 'word_embedder': we, 
                 'device': device,
                 'faith': array([c['f1'] for c in best['column_wise']])}
    torch.save(all_stuff, os.path.join(SAVE_PATH, model_name))


if __name__ == "__main__":
    train_rnn()