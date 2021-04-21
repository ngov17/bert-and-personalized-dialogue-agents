from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import json
import random
#from pytorch_pretrained_bert import cached_path
from transformers import BertTokenizer, BertModel
from itertools import chain
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertMLMClassifier(nn.Module):
    def __init__(self, dropout=0.1, mlm_coeff=1, class_coeff=1, num_choices=2):
        super(BertMLMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # hyper params
        self.num_choices = num_choices
        self.mlm_coeff = mlm_coeff
        self.class_coeff = class_coeff

        # Classification Head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # MLM head
        self.mlm = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

        # loss functions
        self.class_loss = nn.CrossEntropyLoss()
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)    # -1 is mask_token_idx

    def forward(self, input_ids, attention_mask=None, class_labels=None, mlm_labels=None):
        num_choices = input_ids.size(1)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(flat_input_ids, attention_mask=flat_attention_mask)
        output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        class_logits = logits.view(-1, self.num_choices) # bsz x num_choices

        # MLM head:
        mlm_out = output[0::self.num_choices]    # bsz x seq_len x hidden_size
        mlm_logits = self.mlm(mlm_out)  #bsz x seq_len x vocab_size

        if mlm_labels is not None:
            class_loss = self.class_loss(class_logits, class_labels)
            mlm_loss = self.mlm_loss(mlm_logits.transpose(1, 2), mlm_labels)
            return self.mlm_coeff*mlm_loss + self.class_coeff*class_loss, class_logits, mlm_logits
        else:
            return class_logits, mlm_logits

    def get_embeddings(self, input, attention_mask):
        out = self.bert(input, attention_mask=attention_mask)
        return out.last_hidden_state  # bsz x seq_len x hidden_size


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len, pretrained_bert=None, mlm=True, bare=False):
        super(TransformerLM, self).__init__()

        self.transformer_sz = 768
        self.n_heads = 4
        self.d_ff = self.transformer_sz * 4
        self.mlm = mlm

        self.bare = bare

        self.loss = nn.CrossEntropyLoss(ignore_index=0)    # 0 is pad_token_idx

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=768)
        if pretrained_bert is None:
            #self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=768)
            self.pretrained = False
        else:
            self.embeddings_bert = pretrained_bert
            self.pretrained = True


        self.encoder = nn.TransformerEncoderLayer(d_model=self.transformer_sz, nhead=self.n_heads, dim_feedforward=self.d_ff)
        self.linear = nn.Linear(self.transformer_sz, vocab_size)

        def generate_mask(sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
        self.mask = generate_mask(seq_len).to(DEVICE)





    def forward(self, inputs, labels, attention_mask=None):

        if self.pretrained:
            # learned embeddings:
            emb = self.embeddings(inputs)# bsz x seq_len x hidden_size
            #with torch.no_grad():
            if self.mlm:
                #print(self.embeddings.get_embeddings(inputs, attention_mask))
                with torch.no_grad():
                    bert_emb = self.embeddings_bert.get_embeddings(inputs, attention_mask)
                emb = emb + bert_emb
            else:
                if self.bare:
                    with torch.no_grad():
                        bert_emb = self.embeddings_bert(inputs, attention_mask=attention_mask, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        bert_emb = self.embeddings_bert(inputs.unsqueeze(1), attention_mask=attention_mask, output_hidden_states=True)
                emb = emb + bert_emb.hidden_states[0]  # embeddings of the bare Multiple Choice Model
        else:
            emb = self.embeddings(inputs)   # bsz x seq_len x hidden_size

        encoded = self.encoder(emb.transpose(0, 1), src_mask=self.mask) #seq_len x bsz x hidden_size
        encoded = encoded.transpose(0, 1)   # bsz x seq x hidden
        logits = self.linear(encoded)   # bsz x seq_len x vocab_size

        loss = self.loss(logits.transpose(1, 2), labels)

        return loss, logits

