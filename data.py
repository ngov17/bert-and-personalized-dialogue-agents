from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
import urllib.request
import json
import random
# from pytorch_pretrained_bert import cached_path
from transformers import BertTokenizer, BertModel
from itertools import chain
import torch
from models import BertMLMClassifier

url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

# additional tokens:
# not added to tokenizer vocabulary, used in preprocessing to discard
# persona info
PERSONA = "persona:"
# added to vocabulary
BOS = "<BOS/>"
SEP = "<SEP/>"
EOS = "<EOS/>"

# mask value for inputs to GPT-LM Head Model
mask_val = -100
personalities = []


class BertClassifierDataset(Dataset):
    """Chatbot dataset."""

    def __init__(self, tokenizer, seq_len, type="train", persona=True, flip_reply=True, mlm=False, lm=False, personality_classifier=False, gpt2=False):
        """
        Args:

        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.lm = lm
        self.personality_classifier = personality_classifier
        if self.personality_classifier:
            print("personality is being classified")
        else:
            print("response is classified")
        # special tokens
        self.pad_token_idx = tokenizer.pad_token_id
        self.mask_token_idx = tokenizer.mask_token_id
        self.cls_token_idx = tokenizer.cls_token_id
        self.sep_token_idx = None
        # preprocessing
        self.inputs = []
        self.labels = []
        self.att_masks = []
        self.labels_mlm = []
        self.lengths = []

        # inputs in form [[persona, history, reply], [persona, history,
        # distractor]]
        self.raw_inputs = get_raw_data(type)
        x = 25000 if type == "train" else 5000
        self.personalities = personalities
        print(len(personalities))
        if lm:
            for raw in self.raw_inputs[:x]:
                inp = raw[0]
               # if len(self.foldr(inp[0])) > 512 or len(self.foldr(inp[1])) > 512 or len(inp[2]) > 512:
                # break
                tokenized_persona, tokenized_history, tokenized_reply = \
                    tokenizer(self.foldr(inp[0]))["input_ids"], tokenizer(self.foldr(inp[1]))["input_ids"], \
                    tokenizer(inp[2])["input_ids"]
                input, _ = self.pad(
                    tokenized_persona + tokenized_history[1:] + tokenized_reply[1:][:-1])
                label, _ = self.pad(self.mask(
                    tokenized_persona) + self.mask(tokenized_history[1:]) + tokenized_reply[2:], True)
                att_mask = [1 for _ in range(
                    len(tokenized_persona + tokenized_history[1:]))]
                att_mask += [0 for _ in range(self.seq_len - len(att_mask))]
                if len(att_mask) > self.seq_len:
                    att_mask = att_mask[0:self.seq_len]
                self.att_masks.append(att_mask)
                self.inputs.append(input)
                self.labels.append(label)
                self.lengths.append(len(tokenized_reply[2:]))
            assert len(self.inputs) == len(self.att_masks)
        else:
            for raw in self.raw_inputs[:x]:
                if self.personality_classifier:
                    inp = raw[0]
                    tokenized_persona, tokenized_history, tokenized_reply = \
                        tokenizer(self.foldr(inp[0]))["input_ids"], tokenizer(self.foldr(inp[1]))["input_ids"], \
                        tokenizer(inp[2])["input_ids"]
                    random_persona = self.personalities[
                        random.randrange(len(self.personalities))]
                    random_persona = tokenizer(self.foldr(random_persona))[
                        "input_ids"]
                    # random_persona_2 = self.personalities[
                    #   random.randrange(len(self.personalities))]
                   # random_persona_2 = tokenizer(self.foldr(random_persona_2))[
                   #     "input_ids"]
                    if gpt2:
                        tokenized_persona.append(self.cls_token_idx)
                        random_persona.append(self.cls_token_idx)
                        tokenized_inp, att_mask = self.pad(
                            tokenized_history + tokenized_reply + tokenized_persona)
                        distractor_inp, distractor_mask = self.pad(
                            tokenized_history + tokenized_reply + random_persona)
                        # distractor_inp_2, distractor_mask_2 = self.pad(
                        # tokenized_history + tokenized_reply +
                        # random_persona_2)
                    else:
                        tokenized_inp, att_mask = self.pad(
                            tokenized_reply + tokenized_history[1:] + tokenized_persona[1:])
                        distractor_inp, distractor_mask = self.pad(
                            tokenized_reply + tokenized_history[1:] + random_persona[1:])
                       # distractor_inp_2, distractor_mask_2 = self.pad(
                        # tokenized_reply + tokenized_history[1:] +
                        # random_persona_2[1:])

                    inps = [tokenized_inp, distractor_inp]
                    masks = [att_mask, distractor_mask]
                    self.inputs.append(inps)
                    self.att_masks.append(masks)
                else:
                    inps = []
                    masks = []
                    for i, inp in enumerate(raw):
                        if len(self.foldr(inp[0])) > 512 or len(self.foldr(inp[1])) > 512 or + len(inp[2]) > 512:
                            break
                        tokenized_persona, tokenized_history, tokenized_reply = \
                            tokenizer(self.foldr(inp[0]))["input_ids"], tokenizer(self.foldr(inp[1]))["input_ids"], \
                            tokenizer(inp[2])["input_ids"]
                        if mlm and i == 0:
                            labels, _ = self.mask_sentence(
                                tokenized_persona, type)

                        tokenized_inp, att_mask = (self.pad(tokenized_persona + tokenized_reply[1:] + tokenized_history[1:])
                                                   if flip_reply else self.pad(tokenized_persona + tokenized_history[1:] + tokenized_reply[1:])) \
                            if persona else self.pad(tokenized_history + tokenized_reply[1:])

                        if mlm and i == 0:
                            assert len(att_mask) == self.seq_len
                            tokenized_lab, _ = (
                                self.pad(labels +
                                         self.mask(tokenized_reply[1:]) + self.mask(tokenized_history[1:]))
                                if flip_reply else self.pad(self.mask(tokenized_persona) + self.mask(tokenized_history[1:]) + labels)) \
                                if persona else self.pad(self.mask(tokenized_history) + labels)
                            assert len(tokenized_lab) == self.seq_len
                            self.labels_mlm.append(tokenized_lab)

                        if gpt2:
                            tokenized_reply.append(self.cls_token_idx)
                            tokenized_inp, att_mask = self.pad(
                                tokenized_persona + tokenized_history + tokenized_reply)
                        inps.append(tokenized_inp)
                        masks.append(att_mask)

                    if len(inps) == 2 and len(masks) == 2:
                        self.inputs.append(inps)
                        self.att_masks.append(masks)

            # construct classification labels
            for inp in self.inputs:
                #print(len(inp))
                p = random.random()
                if p < 0.5:
                    # swap inp[0] and inp[1], label is now index 1
                    temp = inp[0]
                    inp[0] = inp[1]
                    inp[1] = temp
                    self.labels.append(1)
    #            elif p < 0.6:
   #                 temp = inp[0]
  #                  inp[0] = inp[2]
 #                   inp[2] = temp
#                    self.labels.append(2)
                else:
                    # default case, label is first index (index 0)
                    self.labels.append(0)

            if not mlm or self.personality_classifier:
                self.labels_mlm = self.labels

            assert len(self.inputs) == len(self.labels) and len(self.inputs) == len(self.att_masks) and len(
                self.inputs) == len(self.labels_mlm)

        print(len(self.inputs))
        print(len(self.att_masks))
        print(len(self.labels))
        print(len(self.labels_mlm))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if not self.lm:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.att_masks[idx]), torch.tensor(self.labels[idx]), torch.tensor(self.labels_mlm[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx]), torch.tensor(self.lengths[idx]), torch.tensor(self.att_masks[idx])

    # Utility Functions
    def foldr(self, input):
        out = ""
        space = " "
        for i in input:
            out += i
            out += space
        return out[:-1]

    def pad(self, input, lm=False):
        att_mask = [1 for _ in range(self.seq_len)]

        idx = self.pad_token_idx

        if len(input) < self.seq_len:
            l_inp = len(input)
            input += [idx] * (self.seq_len - len(input))
            for j in range(l_inp, self.seq_len):
                att_mask[j] = 0
        elif len(input) > self.seq_len:
            input = input[0:self.seq_len]
        return input, att_mask

    def mask(self, sentence):
        if self.lm:
            idx = self.pad_token_idx
        else:
            idx = self.mask_token_idx
        out = [idx for _ in sentence]
        return out

    def mask_sentence(self, sentence, type):
        labels = [self.mask_token_idx for _ in sentence]
        if type == "valid":
            return labels, None
        # stores the actual number of tokens being predicted in each sentence,
        # used for perplexity calc.
        lens = 0
        for i, token_id in enumerate(sentence):
            prb = random.random()
            # mask 15% of tokens
            if prb < 0.15:
                # if token is masked, label should be un-masked (Masked
                # Language Modelling)
                labels[i] = token_id
                # increment lens by 1, because this token is predicted
                lens += 1
                prb_mask = prb / 0.15
                # for each token to be masked, mask 80% of time, replace with random token 10% of the time,
                # and keep the same token 10% of time (according to paper)
                if prb_mask < 0.8:
                    # Replace token with <mask/> token 80% of the time
                    sentence[i] = self.mask_token_idx
                elif prb_mask < 0.9:
                    sentence[i] = random.randrange(
                        len(self.tokenizer.get_vocab()))
        return labels, lens


def get_raw_data(type):
    """

    :param type: "train" or "test"

    :return: raw inputs
    """

    # Download and load JSON dataset
    print("fetching json data...")
    # personachat_file = cached_path(url)
    # with open(personachat_file, "r", encoding="utf-8") as f:
    #    dataset = json.loads(f.read())
    with urllib.request.urlopen(url) as u:
        dataset = json.loads(u.read().decode())

    t_inputs = []
    for elem in dataset[type]:
        persona = elem["personality"]
        personalities.append(persona)
        for utterance in elem["utterances"]:
            history = utterance["history"]
            # true reply, random distractor
            candidates = [utterance["candidates"]
                          [-1], utterance["candidates"][-2]]
            input = []
            for c in candidates:
                inps = [persona, history]
                inps.append(c)
                input.append(inps)
            t_inputs.append(input)
    print(type + " raw data processed")
    return t_inputs


def load_dataset(type, tokenizer, batch_size, seq_len, persona=True, flip=True, mlm=False, lm=False, per_class=False, gpt2=False):
    """
        Loads dataset

    """
    if lm:
        dataset = BertClassifierDataset(tokenizer, seq_len, type=type, lm=True)
    else:
        print("here")
        dataset = BertClassifierDataset(
            tokenizer, seq_len, type=type, persona=persona, flip_reply=flip, mlm=mlm, personality_classifier=per_class, gpt2=gpt2)
    print(len(dataset))    
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# d = BertClassifierDataset(tokenizer, 100, mlm=True)
# loader = DataLoader(d, 32, shuffle=True)
# model = BertMLMClassifier()
# for inp, mask, lab, mlm_lab in loader:
#     print(mlm_lab.size())
#     loss = model(inp, mask, lab, mlm_lab)
#     print(loss)
#     exit(0)
