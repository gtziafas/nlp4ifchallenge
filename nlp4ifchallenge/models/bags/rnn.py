from ..types import *
from ..preprocessing import read_labeled
from ..utils.embeddings import make_word_embedder

import torch
from torch import cat, stack
from torch.nn import GRU, Dropout
from adabelief_pytorch import AdaBelief


class MultiLabelRNN(Module):
    def __init__(self, num_classes: int, 
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


def default_rnn():
    return MultiLabelRNN(7, 300, 150, 1, 0.25)


def collator(batch: List[array, array]) -> Tuple[Tensor, Tensor]:
    xs, ys = zip(*batch)
    xs = pad_sequence([tensor(x, dtype=torch.float) for x in xs], batch_first=True)
    ys = stack([tensor(y, dtype=torch.long) for y in ys], dim=0)
    return xs, ys 


def main(train_path: str, 
        dev_path: str, 
        test_path: str,
        batch_size: int = 16, 
        num_epochs: int = 100,
        val_feq: int = 5,
        embeddings: str = 'glove_lg',
        device: str = 'cuda'):

    # random
    torch.manual_seed(0)

    # data
    train_data = read_labeled(train_path)
    dev_data = read_labeled(dev_path)

    # representation
    we = make_word_embedder(embeddings)
    train_data = [(we(sent), label) for sent, label in train_data]
    dev_data = [(we(sent), label) for sent, label in dev_data]
    train_dl = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=collator)
    dev_dl = DataLoader(dev_data, shuffle=True, batch_size=batch_size, collate_fn=collator)

    # model, optim, loss 
    model = default_rnn().to(device)
    optim = AdaBelief(model.parameters(), lr=1e-03, weight_decay=1e-02)
    loss_fn = BCEWithLogitsLoss(reduction='mean').to(device)

    for epoch in range(num_epochs):
        bce_loss, accu, hamm_loss = train_epoch(model, train_dl, optim, loss_fn, device)
        print(f'Epoch {epoch+1}/{num_epochs}: BCE={bce_loss:.4f}\tAccuracy={accu*100:2.2f}%\tHamming={hamm_loss*100:2.2f}%')
        if epoch  % val_feq == 0:
            dev_bce_loss, dev_accu, dev_hamm_loss = eval_epoch(model, dev_dl, loss_fn, device)
            print('=' * 96)
            print(f'EVALUATION: BCE={dev_bce_loss:.4f}\tAccuracy={dev_accu*100:2.2f}%\tHamming={dev_hamm_loss*100:2.2f}%'

