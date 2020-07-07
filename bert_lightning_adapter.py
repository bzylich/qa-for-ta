from typing import Dict
from collections import OrderedDict
from functools import partial

import lineflow as lf
import lineflow.datasets as lfds
import lineflow.cross_validation as lfcv
from lineflow.core import lineflow_load

import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from adapter_bert import AdapterBert, setup_adapters
import os

# model_name = "bert-large-uncased-whole-word-masking"
model_name = "bert-base-uncased"


cache_directory = "/mnt/nfs/work1/andrewlan/bzylich/cached_models/"

nb_gpus = 1
nb_nodes = 1
# world = nb_gpus * nb_nodes

long_candidates_per_example = 4
nq_save_path = "/mnt/nfs/work1/andrewlan/bzylich/datasets/natural_questions/simplified/v1.0-simplified-nq-train-" + str(long_candidates_per_example) + "-cands-per-example.pkl"

MAX_LEN = 300
NUM_LABELS = 2


def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:
    # Given two sentences, x["string1"] and x["string2"], this function returns BERT ready inputs.
    inputs = tokenizer.encode_plus(
            x["question"],
            x["context"],
            add_special_tokens=True,
            max_length=MAX_LEN,
            )

    # First `input_ids` is a sequence of id-type representation of input string.
    # Second `token_type_ids` is sequence identifier to show model the span of "string1" and "string2" individually.
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    # BERT requires sequences in the same batch to have same length, so let's pad!
    padding_length = MAX_LEN - len(input_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Super simple validation.
    assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
    assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
    assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

    # Convert them into PyTorch format.
    label = torch.tensor(int(not x["is_impossible"])).long()
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    # DONE!
    return {
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
            }


def get_dataloader():
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case="uncased" in model_name, cache_dir=cache_directory)
    preprocessor = partial(preprocess, tokenizer)

    train = lfds.Squad("train", 2)
    test = lfds.Squad("dev", 2)

    train, val = lfcv.split_dataset_random(train, int(len(train) * 0.9), seed=42)
    train = train.map(preprocessor)
    print("SQuAD Train dataset length:", len(train), flush=True)
    # nq = lineflow_load(nq_save_path)
    # train, val = lfcv.split_dataset_random(nq, int(len(nq) * 0.9), seed=42)
    # train = train.map(preprocessor)
    # nq = nq.map(preprocessor)
    # print("NQ Train dataset length:", len(nq), flush=True)
    # train = ConcatDataset([train, nq])

    print("Train dataset length:", len(train), flush=True)
    print("Val dataset length:", len(val), flush=True)
    print("Test dataset length:", len(test), flush=True)

    val = val.map(preprocessor)
    test = test.map(preprocessor)

    return train, val, test


class Model(pl.LightningModule):

    def __init__(self, num_epochs, adapter_size, learning_rate, batch_size):
        super(Model, self).__init__()
        self.num_epochs = num_epochs
        self.adapter_size = adapter_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        setup_adapters(self.adapter_size)
        model = AdapterBert.from_pretrained(model_name, num_labels=NUM_LABELS, cache_dir=cache_directory)
        self.model = model

        train, val, test = get_dataloader()
        self._train = train
        self._val = val
        self._test = test
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]  # ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0  # 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8
        )

        t_total = len(self.train_dataloader()) * self.num_epochs
        t_warmup = int(0.1 * float(t_total))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_warmup, num_training_steps=t_total)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
            })

        return output

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            correct_count = correct_count.unsqueeze(0)

        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
            })
        return output

    def validation_end(self, outputs):
        val_acc = sum([torch.mean(out["correct_count"]) if (self.trainer.use_dp or self.trainer.use_ddp2) else out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([torch.mean(out["val_loss"]) if (self.trainer.use_dp or self.trainer.use_ddp2) else out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })

        return output

    def test_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result

    @pl.data_loader
    def train_dataloader(self):
        if self._train_dataloader is None:
            train_dist_sampler = torch.utils.data.distributed.DistributedSampler(self._train)
            self._train_dataloader = DataLoader(
                self._train,
                sampler=train_dist_sampler,  # RandomSampler(train),
                # sampler=RandomSampler(train),
                batch_size=self.batch_size,
                num_workers=0
            )
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        if self._val_dataloader is None:
            val_dist_sampler = torch.utils.data.distributed.DistributedSampler(self._val)
            self._val_dataloader = DataLoader(
                self._val,
                sampler=val_dist_sampler,  # SequentialSampler(val),
                # sampler=SequentialSampler(val),
                batch_size=self.batch_size,
                num_workers=0
            )
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        if self._test_dataloader is None:
            test_dist_sampler = torch.utils.data.distributed.DistributedSampler(self._test)
            self._test_dataloader = DataLoader(
                self._test,
                sampler=test_dist_sampler,  # SequentialSampler(test),
                # sampler=SequentialSampler(test),
                batch_size=self.batch_size,
                num_workers=0
            )
        return self._test_dataloader


if __name__ == "__main__":
    # BEGIN Hyperparameters
    BATCH_SIZE = 32

    # LEARNING_RATE = 5e-5
    # ADAPTER_SIZE = 8
    # NUM_EPOCHS = 3
    # END
    learning_rate_options = [1e-3, 1e-2, 1e-1]
    adapter_size_options = [64, 256]
    num_epochs_options = [20, 30]

    for NUM_EPOCHS in num_epochs_options:
        for LEARNING_RATE in learning_rate_options:
            for ADAPTER_SIZE in adapter_size_options:
                name = model_name + "_" + str(LEARNING_RATE) + "_" + str(BATCH_SIZE) + "_adapter_" + str(ADAPTER_SIZE) + "_epochs_" + str(NUM_EPOCHS) + "_SQuAD_Grid"  # _NQ_2-cands-per-example_plus-
                ckpt_directory = "./" + name + "/"
                SAVE_PREFIX = name + "_"

                # early_stop_callback = EarlyStopping(
                #     monitor="val_loss",
                #     min_delta=0.0,
                #     patience=3,
                #     verbose=True,
                #     mode="min"
                # )

                # default logger used by trainer
                logger = TestTubeLogger(
                    save_dir="./lightning_logs/",
                    name=name,
                    debug=False,
                    create_git_tag=False
                )

                checkpoint_callback = ModelCheckpoint(
                    filepath=ckpt_directory,
                    save_top_k=1,
                    verbose=True,
                    monitor='val_loss',
                    mode='min',
                    prefix=SAVE_PREFIX
                )

                trainer = pl.Trainer(
                    logger=logger,
                    nb_gpu_nodes=nb_nodes,
                    gpus=nb_gpus,
                    early_stop_callback=False,  # early_stop_callback,
                    checkpoint_callback=checkpoint_callback,
                    distributed_backend='ddp',
                    amp_level='O2',
                    use_amp=True,
                    max_epochs=NUM_EPOCHS
                )

                print("-----------------------------------------------------------------------")
                print(name)
                model = Model(NUM_EPOCHS, ADAPTER_SIZE, LEARNING_RATE, BATCH_SIZE)
                print("all params:", model.model.count_all_params(), "trainable params:", model.model.count_trainable_params())

                print("Model initiated.", flush=True)

                print("Commencing training...", flush=True)
                trainer.fit(model)
                print("Training completed.", flush=True)
                # print("Testing model...", flush=True)
                # trainer.test()
                # print("Finished!", flush=True)

                del logger
                del checkpoint_callback
                del trainer
                del model
                torch.cuda.empty_cache()
