#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
TOKENIZERS_PARALLELISM=false python splice-text.py
'''

# Standard library imports.
import random
import sys

# Pytorch imports.
import torch
import pytorch_lightning as pl

# Transformer imports
import transformers

# Ranger (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
import ranger

#https://github.com/jettify/pytorch-optimizer#radam
#from torch_optimizer import RAdam



class TokenizedTextDataModule(pl.LightningDataModule):
   '''A simple data class for tokenized text.'''
   def __init__(
         self,
         model_path:       str,
         data_path:        str,
         max_seq_length:   int = 512,
   ):
      super().__init__()
      self.model_path = model_path
      self.data_path = data_path
      self.max_seq_length = max_seq_length
      self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
            self.model_path, use_fast=True)

   def trim(self, tokens):
      if len(tokens) > self.max_seq_length:
         tokens = tokens[:self.max_seq_length]
         tokens[self.max_seq_length-1] = self.tokenizer.eos_token_id
      return tokens

   def setup(self, stage):
      '''Create a dataset from texts for a guessing task. The texts
      consist of 4 types of texts: i) the original texts, ii) texts
      where heads are swapped, i.e., they contain the beginning of a
      text and the end of another at random, iii) and iv) 
      head-to-tail swaps of i) and ii). Texts i) and iii) have label
      0 and texts ii) and iv) have label 1, so the task is to
      discover whhich texts contain information from two different
      documents.'''
      # Read-in the tokens as torch long (original texts).
      data = list()
      with open(self.data_path) as f:
         for line in f:
            tokens = self.trim([int(x) for x in line.split()])
            data.append((torch.tensor(tokens), 0))
      # Swap texts head-to-tail.
      random.shuffle(data)
      stopid, = self.tokenizer.encode('.', add_special_tokens=False)
      for i in range(len(data)):
         AB,_ = data[i]
         # Cut on a full stop in the middle.
         whereAB, = torch.where(AB == stopid)
         if len(whereAB) < 1:
            continue # Cannot swap realisticaly.
         cutAB = whereAB[len(whereAB)//2]
         # Swap head and tail (except first and last tokens).
         BA = torch.cat((AB[:1], AB[cutAB:-1], AB[1:cutAB], AB[-1:]))
         data.append((BA, 0))
      # Splice texts in random pairs.
      for i in range(0, len(data), 2):
         X,_ = data[i]
         Y,_ = data[i+1]
         # Cut both on a full stop in the middle.
         whereX, = torch.where(X == stopid)
         whereY, = torch.where(Y == stopid)
         if len(whereX) < 2 or len(whereY) < 2:
            continue # Cannot splice realisticaly.
         cutX = whereX[len(whereX)//2]
         cutY = whereY[len(whereY)//2]
         # Swap heads (or tails).
         XY = self.trim(torch.cat((X[:cutX], Y[cutY:])))
         YX = self.trim(torch.cat((Y[:cutY], X[cutX:])))
         data.append((XY, 1))
         data.append((YX, 1))
      random.shuffle(data)
      self.train_data = data[len(data)//20:]
      self.test_data = data[:len(data)//20]

   def collate(self, examples):
      '''Pad batch and return pair of tensors. This allows the
      data to be kept in emory in a compact non-tensor format
      and to serve a batch in tensor format for the GPUs.'''
      inputs = torch.nn.utils.rnn.pad_sequence(
            [x for x,y in examples],
            batch_first = True,
            padding_value = self.tokenizer.pad_token_id
      )
      labels = torch.tensor([y for x,y in examples])
      return {'input_ids': inputs, 'labels': labels}


class MeaningfulBERT(pl.LightningModule):
   def __init__(
         self,
         model_path:       str,
         num_labels:       int   = 2,
         learning_rate:    float = 1e-4,
         adam_epsilon:     float = 1e-8,
         warmup_steps:     int   = 0,
         weight_decay:     float = 0.0,
         **kwargs
   ):
      super().__init__()
      self.save_hyperparameters()
      self.config = transformers.AutoConfig.from_pretrained(model_path,
            num_labels=num_labels)
      self.model = (transformers.AutoModelForSequenceClassification.
            from_pretrained(model_path, config=self.config))

   def forward(self, **inputs):
      return self.model(**inputs)

   def training_step(self, batch, batch_idx):
      outputs = self(**batch)
      loss = outputs[0]
      return loss

   def configure_optimizers(self):
      "Prepare optimizer and schedule (linear warmup and decay)"
      model = self.model
      no_decay = ["bias", "LayerNorm.weight"]
      optimizer_grouped_parameters = [
          {
              "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": self.hparams.weight_decay,
          },
          {
              "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0,
          },
      ]
      # Original optimizer from Transformers. It works but needs warmup.
      # optimizer = transformers.AdamW(optimizer_grouped_parameters,
      #      lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      # The RAdam optimizer works approximately as well as Ranger.
      #optimizer = RAdam(optimizer_grouped_parameters,
      #      lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      # The Ranger optimizer is the combination of RAdam and Lookahead. It
      # works well for this task. The best conditions seem to be learning
      # rate 1e-4 w/ RAdam or Ranger, gradient accumulation of 2 batches.
      optimizer = ranger.Ranger(optimizer_grouped_parameters,
            lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

      # The constant scheduler does nothing. Replace with another
      # scheduler if required.
      scheduler = transformers.get_constant_schedule(optimizer)
      scheduler = {
          'scheduler': scheduler,
          'interval': 'step',
          'frequency': 1
      }
      return [optimizer], [scheduler]


if __name__ == '__main__':
   pl.seed_everything(123)

   # Load the pre-trained model.
   transformers.logging.set_verbosity_error()
   model = MeaningfulBERT('./BERT-model/checkpoint-100000/')

   # Prepare the data.
   text_data = TokenizedTextDataModule('./tokenizer-model/',
         'head-10-percent.txt')
   text_data.setup('fit')

   # Set up the data loader.
   dataloader = torch.utils.data.DataLoader(
         text_data.train_data,
         batch_size = 20,
         collate_fn = text_data.collate,
         num_workers = 8
   )
   # Set up the trainer and do it. Specify gradient accumulation over
   # two batches, and of course make the best use of your GPUs and
   # the DDP accelerator.
   trainer = pl.Trainer(gpus=4, accelerator='ddp', accumulate_grad_batches=2, max_epochs=1)
   trainer.fit(model, dataloader)
