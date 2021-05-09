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
      'Pad batch and return pair of tensors.'
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
         learning_rate:    float = 1e-5,
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
      optimizer = transformers.AdamW(optimizer_grouped_parameters,
            lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

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
         'encoded-abstracts-all.txt')
   text_data.setup('fit')

   # Set up the data loader.
   dataloader = torch.utils.data.DataLoader(
         text_data.train_data,
         batch_size = 20,
         collate_fn = text_data.collate,
         num_workers = 8
   )
   # Set up the trainer and do it.
   trainer = pl.Trainer(gpus=4, accelerator='ddp', max_epochs=4)
   trainer.fit(model, dataloader)
