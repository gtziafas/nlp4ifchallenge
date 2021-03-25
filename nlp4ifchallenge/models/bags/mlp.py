from ...types import *
from ...preprocessing import read_labeled
from ..utils.embeddings import make_word_embedder
from ..training import train_epoch, eval_epoch
from .tf_idf import extract_tf_idfs

from torch import manual_seed
from adabelief_pytorch import AdaBelief


class MultiLabelMLP(Module, Model):
    def __init__(self, num_classes: int, inp_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden = Linear(in_features=inp_dim, out_features=hidden_dim)
        self.cls = Linear(in_features=hidden_dim, out_features=num_classes)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.cls(x)
        return x 


def default_bags_mlp():
    return MultiLabelMLP(7, 300, 128, 0.33)


def tf_idf_bags_mlp(num_components: int):
    return MultiLabelMLP(7, 300 + num_components, 128, 0.33)


def labels_to_vec(labels: List[Label]) -> array:
    return array([1 if l == True else 0 for l in labels])


def train_mlp(train_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_train.tsv', 
        dev_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_dev_input.tsv', 
        test_path: str = '',
        batch_size: int = 16, 
        num_epochs: int = 100,
        val_feq: int = 5,
        with_tf_idf: int = 100,
        embeddings: str = 'glove_lg',
        device: str = 'cuda'):

    # random
    manual_seed(0)

    # data
    train_data_raw = read_labeled(train_path)
    dev_data_raw = read_labeled(dev_path)

    # bag of embedds representation by average pooling
    we = make_word_embedder(embeddings)
    train_data = [(we(tweet.text).mean(axis=0), labels_to_vec(tweet.labels)) for tweet in train_data_raw]
    dev_data = [(we(tweet.text).mean(axis=0), labels_to_vec(tweet.labels)) for tweet in dev_data_raw]

    # optionally tf-idf representations
    if with_tf_idf:
        train_tf_idfs, lsa, extractor = extract_tf_idfs(text=[tweet.text for tweet in train_data_raw], lsa_components=with_tf_idf)
        train_data = [(np.concatenate((bag, train_tf_idfs[idx]), axis=-1), l) for idx, (bag, l) in enumerate(train_data)]
        # extract features for dev data based on train data transforms
        dev_tf_idfs = lsa.transform(extractor.transform([tweet.text for tweet in dev_data_raw])).astype('float32')
        dev_data = [(np.concatenate((bag, dev_tf_idfs[idx]), axis=-1), l) for idx, (bag, l) in enumerate(dev_data)]

    train_dl = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    dev_dl = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    # model, optim, loss 
    model = default_bags_mlp().to(device) if not with_tf_idf else tf_idf_bags_mlp(with_tf_idf).to(device)
    optim = AdaBelief(model.parameters(), lr=1e-03, weight_decay=1e-02)
    loss_fn = BCEWithLogitsLoss(reduction='mean').to(device)

    train_log, dev_log = [], []
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


if __name__ == "__main__":
    train_mlp()