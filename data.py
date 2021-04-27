import time
import numpy as np
import torchtext
from torchtext.legacy import data
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import tensor2text
class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(iter(self.pos_iter), iter(self.neg_iter)):
            if batch_pos.text.size(0) == batch_neg.text.size(0):
                yield batch_pos.text, batch_neg.text

class EnronIterator(object):
    def __init__(self, iter_1, iter_2):
        self.iter_1 = iter_1
        self.iter_2 = iter_2
    
    def __iter__(self):
        for batch_1, batch_2 in zip(iter(self.iter_1), iter(self.iter_2)):
            if batch_1.text.size(0) == batch_2.text.size(0):
                yield batch_1.text, batch_2.text

def load_enron(config, 
                train_input_1='train_input_1.txt', 
                train_input_2='train_input_2.txt',
                test_input_1='test_input_1.txt', 
                test_input_2='test_input_2.txt'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')

    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_1, train_2 = map(dataset_fn, [train_input_1, train_input_2])
    test_1, test_2 = map(dataset_fn, [test_input_1, test_input_2])

    TEXT.build_vocab(train_1, train_2, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()

        vectors = torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_iter_1, train_iter_2 = map(lambda x: dataiter_fn(x, True), [train_1, train_2])
    test_iter_1, test_iter_2 = map(lambda x: dataiter_fn(x, False), [test_1, test_2])

    train_iters = EnronIterator(train_iter_1, train_iter_2)
    test_iters = EnronIterator(test_iter_1, test_iter_2)

    return train_iters, test_iters, vocab

'''
def load_dataset(config, train_pos='train.pos', train_neg='train.neg',
                 dev_pos='dev.pos', dev_neg='dev.neg',
                 test_pos='test.pos', test_neg='test.neg'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )
    
    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])

    TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors=torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
        
    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_pos_iter, train_neg_iter = map(lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set])
    dev_pos_iter, dev_neg_iter = map(lambda x: dataiter_fn(x, False), [dev_pos_set, dev_neg_set])
    test_pos_iter, test_neg_iter = map(lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set])

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    dev_iters = DatasetIterator(dev_pos_iter, dev_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)
    
    return train_iters, dev_iters, test_iters, vocab

'''
