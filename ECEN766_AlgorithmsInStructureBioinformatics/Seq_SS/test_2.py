from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

print (type(valid_data))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

PAD_IDX = TRG.vocab.stoi['<pad>']
print(PAD_IDX)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
print(INPUT_DIM)
print(OUTPUT_DIM)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

for _, batch in enumerate(train_iterator):

    src = batch.src
    trg = batch.trg

    #print(min(src))
    #print(max(src))
    print(src[0])
    print(len(src[0]))
    print(trg.shape)

    quit()
