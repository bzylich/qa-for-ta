import torch
import torch.nn.functional as F
from transformers.modeling_bert import BertSelfOutput, BertOutput, BertLayerNorm
from transformers import BertForSequenceClassification


class AdapterModule(torch.nn.Module):
    def __init__(self, d_in, adapter_size):
        super().__init__()
        self.project_down = torch.nn.Linear(d_in, adapter_size)
        self.project_up = torch.nn.Linear(adapter_size, d_in)

    def forward(self, x):
        # scale down
        i1 = self.project_down(x)
        # apply nonlinearity
        i2 = F.relu(i1)
        # scale back up
        i3 = self.project_up(i2)
        # apply skip connection
        f = i3 + x

        return f


def setup_adapters(adapter_size):
    def new_bert_output_init(self, config):
        super(BertOutput, self).__init__()
        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.adapter = AdapterModule(config.hidden_size, adapter_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def new_bert_output_forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    BertOutput.__init__ = new_bert_output_init
    BertOutput.forward = new_bert_output_forward

    def new_bert_self_output_init(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.adapter = AdapterModule(config.hidden_size, adapter_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def new_bert_self_output_forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    BertSelfOutput.__init__ = new_bert_self_output_init
    BertSelfOutput.forward = new_bert_self_output_forward


class AdapterBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        total_params = self.count_all_params()
        before = self.count_trainable_params()
        # self.log_all_params()
        self.freeze_for_train()

        trainable_params = self.count_trainable_params()
        print("total params:", total_params)
        print("trainable_params (before):", before, "(after):", trainable_params)
        # self.log_trainable_params()

    def freeze_for_train(self):
        should_train = ["LayerNorm", "classifier", "adapter"]
        for n, p in self.named_parameters():
            if not any(x in n for x in should_train):
                p.requires_grad = False

    def log_all_params(self):
        print(list(n for n, p in self.named_parameters()))

    def log_trainable_params(self):
        print(list(n for n, p in self.named_parameters() if p.requires_grad))

    def count_all_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    cache_directory = "./cache/"  # "/mnt/nfs/work1/andrewlan/bzylich/cached_models/"
    NUM_LABELS = 2
    ADAPTER_SIZE = 8
    setup_adapters(ADAPTER_SIZE)
    test_bert = AdapterBert.from_pretrained(model_name, num_labels=NUM_LABELS, cache_dir=cache_directory)
