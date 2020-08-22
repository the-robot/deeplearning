# ====== IMPORTS ======
import torch
import torch.nn as nn
import torch.optim as optim

import spacy

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k  # german to english dataset
from torchtext.data import Field, BucketIterator

"""
python -m spacy download de
python -m spacy download en
"""


# ====== DATA PREPROCESSING ======
# iNSTALL SPACY
spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")

# Tokenizer
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

# GET TRAIN, VALID, TEST DATASET
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german, english)
)

# build vocabulary
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)



# ====== CREATE TRANSFORMAT NETWORK ======
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx


    def make_src_mask(self, src):
        # to skip some computation

        # src shape: (src_len, N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # output: (N, src_len)

        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask,
        )

        out = self.fc_out(out)
        return out



# ====== SETUP TRAINING ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32


# Modal hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)

# emdeeding and heads are values defined in "attention is all you need" paper
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3

# dropout is set a bit lower in Seq-Seq model than using Fully-Connected
dropout = 0.10

# the sentence length, in this case you don't have src (german) or trg (english)
# length that is 100
# this is used for positional embedding
#
# I.e., if we have sentence longer than 100, we would have to delete them or
#        increase the max_len
max_len = 100

forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]


# Tensorbord for nice plots
writer = SummaryWriter("runs/loss_plot")
step = 0


# Create iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,  # just for efficiency (https://youtu.be/M6adRGJe5cQ?t=1211)
    sort_key = lambda x: len(x.src),
    device = device
)


# Create model
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

# create optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# when we create loss function, we need to ignore padding index
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


# A horse walks under a bridge next to a boat.
sentence = "ein pferd geht unter einer brücke neben einem boot."


for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=max_len
    )

    print(f"Translated example sentence \n {translated_sentence}")
    model.train()


    # start training
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        #     remove the last one because
        #     we want the target that we send in and corresponding output
        #     from that target in the transformer, we want there to be a shift
        #     so the output is one time step or one element ahead of the input of that
        #     corresponding time step
        #
        #     so esentially when we send in the first element of the input to be the
        #     start token, we wanted the first output from the transformer to correspond
        #     to the second element in the target
        #
        #     https://youtu.be/M6adRGJe5cQ?t=1530
        output = model(inp_data, target[:-1])

        # if we have a batch, for every example in the batch, we have
        # predicted translated sentence of let's say 50 words, and for each
        # of those words we have another dimension in the output equal to the
        # target vocabulary and this corresponds to the probability for each word
        # in the vocabulary.
        #
        # so to make cross entrophy works in this case, we gonna have to concatenate
        # the batch of the examples with the words and then keep the last dimension
        # which is the target vocabulary size
        #
        # https://youtu.be/M6adRGJe5cQ?t=1590
        output = output.reshape(-1, output.shape[2])

        # we want the shift of 1 between the output and the target
        # and want the output to be one ahead of the target
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        # max_norm to avoid exploding gradient problems
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1



# ====== CALCULATE BLEU SCORE AFTER TRAINING ======
score = bleu(test_data, model, german, english, device)
print(f"Blue score {score * 100:.2f}")
