from comet_ml import Experiment
import torch
import torch.nn.functional as F
#from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import math
import numpy as np
from data import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMultipleChoice
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from models import BertMLMClassifier, TransformerLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT = False
hyper_params = {
    "batch_size": 8,
    "num_epochs": 1,
    "learning_rate": 3e-5,
    "seq_len": 100
}

def train_lm(model, train_loader, optimizer, experiment):
    model = model.train()
    print("NUM BATCHES: ", len(train_loader))
    with experiment.train():
        for i in range(hyper_params["num_epochs"]):
            for j, (inputs, labels, lens, att_mask) in enumerate(train_loader):
                if BERT:
                    att_mask = att_mask.to(DEVICE)
                else:
                    att_mask = None
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                loss, _ = model(inputs, labels, att_mask)
                print("STEP: ", j)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

def test_lm(model, test_loader, experiment):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """

    model = model.eval()

    total_loss = 0
    total_words = 0

    with experiment.validate():
        for inputs, labels, lens, att_mask in test_loader:
            if BERT:
                att_mask = att_mask.to(DEVICE)
            else:
                att_mask = None
            inputs, labels, lens = inputs.to(DEVICE), labels.to(DEVICE), lens.to(DEVICE)
            with torch.no_grad():
                avg_loss, _ = model(inputs, labels, att_mask)
            print(avg_loss)
            no_tokens = sum(lens)
            total_loss += avg_loss * no_tokens
            total_words += no_tokens
        avg_loss = total_loss / total_words  # avg loss over entire val set
        perplexity = torch.exp(avg_loss).item()
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def train(model, train_loader, optimizer, experiment, mlm, gpt2):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """

    # TODO: Write the training loop here, save trained model weights if needed
    model = model.train()
    with experiment.train():
        if mlm:
            for i in range(hyper_params["num_epochs"]):
                for inputs, att_masks, labels, mlm_labels in train_loader:
                    inputs, att_masks, labels, mlm_labels = inputs.to(DEVICE), att_masks.to(DEVICE), labels.to(DEVICE), mlm_labels.to(DEVICE)
                    loss, _, _ = model(inputs, att_masks, labels, mlm_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        elif gpt2:
            for i in range(hyper_params["num_epochs"]):
                for inputs, att_masks, labels, _ in train_loader:
                    inputs, att_masks, labels = inputs.to(DEVICE), att_masks.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs, mc_labels=labels, att_mask=att_masks)
                    loss = outputs.mc_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        else:
            for i in range(hyper_params["num_epochs"]):
                for inputs, att_masks, labels, _ in train_loader:
                    inputs, att_masks, labels = inputs.to(DEVICE), att_masks.to(DEVICE), labels.to(DEVICE)
                    
                    outputs = model(input_ids=inputs, attention_mask=att_masks, labels=labels)
                    loss = outputs.loss
                    #print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


def accuracy(logits, labels):
    """
    :logits: bsz x num_choices
    """
    total = logits.size(0)  # batch_size
    correct = 0
    for i, logit in enumerate(logits):
        pred = torch.argmax(logit)
        correct += 1 if pred == labels[i] else 0
    return correct, total


def test(model, test_loader, experiment, mlm, gpt2):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """

    model = model.eval()

    # for calculating hits@1 -> accuracy of the model classifying the gold response from the distractor
    total_correct = 0
    total = 0

    with experiment.validate():
        if mlm:
            for inputs, att_masks, labels, mlm_labels in test_loader:
                inputs, att_masks, labels, mlm_labels = inputs.to(DEVICE), att_masks.to(DEVICE), labels.to(DEVICE), mlm_labels.to(DEVICE)
                # during testing and inference, inputs are not masked
                with torch.no_grad():
                    logits, _ = model(inputs, att_masks) # here logits represent the class logits
                correct, tot = accuracy(logits, labels)
                total_correct += correct
                total += tot
        elif gpt2:
            for inputs, att_masks, labels, _ in test_loader:
                inputs, att_masks, labels = inputs.to(DEVICE), att_masks.to(DEVICE), labels.to(DEVICE)
                # during testing and inference, inputs are not masked                                                                                                                                     
                with torch.no_grad():
                    outputs = model(inputs, attention_mask=att_masks, mc_labels=labels)
                logits = outputs.mc_logits
                correct, tot = accuracy(logits, labels)
                total_correct += correct
                total += tot
        else:
            for inputs, att_masks, labels, _ in test_loader:
                print(inputs.size())
                print(att_masks.size())
                inputs, att_masks, labels = inputs.to(DEVICE), att_masks.to(DEVICE), labels.to(DEVICE)
                with torch.no_grad():
                    outputs = model(input_ids=inputs, attention_mask=att_masks, labels=labels)

                # calculate classifcation probabilities using logits
                logits = outputs.logits
                correct, tot = accuracy(logits, labels)
                total_correct += correct
                total += tot

        hits = total_correct / total
        print("hits@1: ", hits)
        experiment.log_metric("hits@1", hits)



def interactive(input, tokenizer, model, personality, top_k=10, ntok=50):
    """
    Generate and print out the response given input using the trained model
    :param input: an input string as prompt (i.e. How are you?)
    :param tokenizer: intialized tokenizer object for encoding the input
    :param model: the trained model to use for generate prediction
    :param top_k: number of samples for top_l sampling
    :param ntok: maximum number of tokens to generate

    Comment: Feed in the input to the model to generate the most probable token
    and concatenate it with current input.
    Continue this process iteratively until the model predicts the padding
    token or reach the maximum number of tokens.
    You may need to add the BOS token and special token to the input sentence
    before passing into model.
    Also, you may want to filter out your input sentence and meaningless tokens
    when printing out the response.
    """
    # TODO: Write the generation function for interacting with trained model
    response = []
    history = tokenizer(input)['input_ids']
    personality = tokenizer(personality)['input_ids']
    input = personality + history[1:]
    
    #input = [tokenizer.bos_token_id] + input + [tokenizer.sep_token_id]
    for i in range(ntok):
        with torch.no_grad():
            i = torch.tensor(input)
            _, logits = model(i)
        #logits = output[0]
        prbs = F.softmax(logits, dim=-1)
        top_k_prbs, top_k_indices = torch.topk(prbs, top_k)
        index = torch.randint(0, top_k, ()).item()
        pred_token = top_k_indices[-1, index]
        if pred_token == tokenizer.sep_token_id or pred_token == tokenizer.cls_token_id:
            break
        response += [pred_token]
        input += [pred_token]
    response = tokenizer.decode(response)
    print("<<  " + response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="BertClassifier or BertMLMClassifier")
    parser.add_argument("-p", "--persona", action="store_true",
                        help="condition on persona")
    parser.add_argument("-b", "--bare_bert", action="store_true",
                        help="Only required when training lm. Indicates a bare bert model is used for embeddings")
    parser.add_argument("-c", "--classifier_only", action="store_true",
                        help="Only required when training lm using saved model. Indicates Bert with only classification head is used.")
    parser.add_argument("-f", "--flip", action="store_true", help="flip reply before history")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="run in interactive mode")
    args = parser.parse_args()

    # load comet-ml experiment
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)
    #experiment = None
    print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("initializing model...")
    # Intialized the pretrained Bert model and optimizer
    lm = False
    gpt2 = False
    if args.model == "BertClassifier":
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased').to(DEVICE)
        mlm = False
        if args.load and not args.bare_bert:
            print("loading saved model..")
            model.load_state_dict(torch.load('model.pt', map_location=DEVICE))
    elif args.model == "BertMLMClassifier":
        model = BertMLMClassifier().to(DEVICE)
        mlm = True
        if args.load and not args.bare_bert:
            print("loading saved model..")
            model.load_state_dict(torch.load('model.pt', map_location=DEVICE))
    elif args.model == "GPT2Classifier":
        print("GPT2 classifier")
        mlm = False
        gpt2 = True
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2').to(DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})
        
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
        if args.load and not args.bare_bert:
            print("loading saved model..")
            model.load_state_dict(torch.load('model.pt', map_location=DEVICE))
    else:
        lm = True
        if args.load:
            BERT = True
            if args.classifier_only:
                bert = BertForMultipleChoice.from_pretrained('bert-base-uncased').to(DEVICE)
            else:
                bert = BertMLMClassifier().to(DEVICE)
            mlm = not args.classifier_only
            print("loading saved model..")
            bert.load_state_dict(torch.load('model.pt', map_location=DEVICE)) 
            model = TransformerLM(len(tokenizer.get_vocab()), 100, bert, mlm=mlm).to(DEVICE)
        elif args.bare_bert:
            BERT = True
            print("loading BERT model..")
            bert = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
            model = TransformerLM(len(tokenizer.get_vocab()), 100, bert, mlm=False, bare=True).to(DEVICE)
        else:
            model = TransformerLM(len(tokenizer.get_vocab()), 100).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    if lm:
        print("loading data...")
        print("loading train data")
        train_loader = load_dataset("train", tokenizer, hyper_params['batch_size'], hyper_params['seq_len'], lm=True)
        print("loading test data")
        test_loader = load_dataset("valid", tokenizer, hyper_params['batch_size'], hyper_params['seq_len'], lm=True)
    else:
        print("loading data...")
        print("loading train data")
        print("flip_reply: ", args.flip)
        print("personality conditioning: ", args.persona)
        train_loader = load_dataset("train", tokenizer, hyper_params['batch_size'], hyper_params['seq_len'], args.persona, args.flip, mlm, per_class=args.load, gpt2=gpt2)
        print("loading test data")
        test_loader = load_dataset("valid", tokenizer, hyper_params['batch_size'], hyper_params['seq_len'], args.persona, args.flip, mlm, per_class=args.load, gpt2=gpt2)
    print("preprocessing over")
    
#     if args.load:
#         model.load_state_dict(torch.load('model.pt', map_location=DEVICE))
    if args.train:
        # run train loop here
        print("running training loop...")
        if lm:
            train_lm(model, train_loader, optimizer, experiment)
        else:
            train(model, train_loader, optimizer, experiment, mlm, gpt2)
    if args.save:
        if lm:
            torch.save(model.state_dict(), 'model_lm.pt')
        else:
            torch.save(model.state_dict(), 'model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        if lm:
            test_lm(model, test_loader, experiment)
        else:
            test(model, test_loader, experiment, mlm, gpt2)
    if args.interactive:
        model.load_state_dict(torch.load('model_lm.pt', map_location=DEVICE))
        personalities = train_loader.dataset.personalities
        ind = random.randrange(len(personalities))
        personality = personalities[ind]
        per = ""
        for p in personality:
            per += p + " "
        per = per[:-1]
        print("SELECTED PERSONALITY: ")
        print(per)
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input(">> ")
            interactive(input_text, tokenizer, model, per)
