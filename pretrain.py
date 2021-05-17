#!/usr/bin/ env python
# -*- coding:utf-8 -*-


'''Simple pre-training pipeline inspired from the Esperanto
tutorial. It is almost obsolete as of mid 2021.'''

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer-model", max_len=512)

from tokenizers import AddedToken
with open('used-symbols.txt') as f:
   genes = [AddedToken(content=line.rstrip(), single_word=True, normalized=False) for line in f]
tokenizer.add_tokens(genes)

from transformers import RobertaConfig
config = RobertaConfig(
    vocab_size = len(tokenizer),
    max_position_embeddings = 512 + 2,
    num_attention_heads = 12,
)
from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./abstracts-all.txt",
    block_size=256,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./BERT-model",
    overwrite_output_dir=True,
    num_train_epochs=35,
    per_device_train_batch_size=25,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./BERT-model")
