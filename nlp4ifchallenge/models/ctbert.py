from .utils.bert_tokenization import *
from transformers import BertModel

import torch
from torch.nn import Linear, BCEWithLogitsLoss, Dropout
from torch.utils.data.dataloader import DataLoader
from adabelief_pytorch import AdaBelief

tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')


def tensorize_labeled(tweets: List[LabeledTweet]) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer) for tweet in tweets]


def tensorize_unlabeled(tweets: List[Tweet]) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer) for tweet in tweets]


class CTBERT(Module, Model):
    def __init__(self):
        super(CTBERT, self).__init__()
        self.core = BertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        self.dropout = Dropout(0.5)
        self.classifier = Linear(1024, 7)

    def forward(self, x: Tensor):
        attention_mask = x.ne(tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    def predict(self, tweets: List[Tweet]) -> List[str]:
        tensorized = pad_sequence(tensorize_unlabeled(tweets), padding_value=tokenizer.pad_token_id)
        preds = self.forward(tensorized).sigmoid().round().long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]


def main(train_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_train.tsv',
         dev_path: str = './nlp4ifchallenge/data/covid19_disinfo_binary_english_dev_input.tsv',
         test_path: str = ''):

    # todo
    torch.manual_seed(0)

    model = CTBERT().cuda()
    model.train()

    train_ds = read_labeled(train_path)
    train_dl = DataLoader(tensorize_labeled(train_ds), batch_size=16,
                          collate_fn=lambda batch: collate_tuples(batch, tokenizer.pad_token_id))

    criterion = BCEWithLogitsLoss()
    optimizer = AdaBelief(model.classifier.parameters(), lr=1e-05, weight_decay=1e-02)

    num_epochs = 5
    for epoch in range(num_epochs):
        for bx, by in train_dl:
            bp = model(bx.cuda())
            batch_loss = criterion(bp, by.float().cuda())
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()