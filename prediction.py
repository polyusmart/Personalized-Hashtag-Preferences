import torch
import argparse
import logging
import json
import time
import numpy as np
import random
from Modules.mlp import Mlp
from itertools import chain
from torch.optim import Adam
import re
from tqdm import tqdm
from utils.scratch_dataset import my_collate
from utils.scratch_dataset import ScratchDataset
import torch.utils.data as data
import torch.nn.functional as F
import argparse
from train import train_mlp
from pykp.model import Seq2SeqModel, NTM
from process_opt import process_opt
from utils.time_log import time_since
from data_loader import load_data_and_vocab
import os
from utils.config import my_own_opts
from utils.config import vocab_opts
from utils.config import model_opts
from utils.config import train_opts
from utils.config import init_logging

CUDA_LAUNCH_BLOCKING=1

torch.manual_seed(2021)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2021)


parser = argparse.ArgumentParser(description='vae_train.py',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
my_own_opts(parser)
vocab_opts(parser)
model_opts(parser)
train_opts(parser)

opt = parser.parse_args()
opt = process_opt(opt)
opt.input_feeding = False
opt.copy_input_feeding = False


opt.device = torch.device("cuda:%d" % opt.gpuid)
print(opt.device)
logging = init_logging(log_file=opt.model_path + '/output.log', stdout=True)
logging.info('Parameters:')
[logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]


def init_optimizers(model, ntm_model, opt):
    optimizer_seq2seq = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_seq2seq, optimizer_ntm, optimizer_whole

def invert_dict(d):
    return dict((v,k) for k,v in d.items())

def main(opt):
    # read files
    with open('./Data/embeddings.json', 'r') as f:
        text_emb_dict = json.load(f)

    with open('./Data/userList.txt', "r") as f:
        x = f.readlines()[0]
        user_list = re.findall(r"['\'](.*?)['\']", str(x))

    train_file = './Data/train.csv'
    valid_file = './Data/valid.csv'
    test_file = './Data/test.csv'
    
    if 1:
        start_time = time.time()
        train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
        word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=True)

        vocab_dict = bow_dictionary

        opt.bow_vocab_size = len(bow_dictionary)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        start_time = time.time()
        model = Seq2SeqModel(opt).cuda()
        ntm_model = NTM(opt).cuda()
        optimizer_seq2seq, optimizer_ntm, optimizer_whole = init_optimizers(model, ntm_model, opt)


        for i in range(1, 2):
            #vae_train_mixture.train_model(i, model, ntm_model, optimizer_seq2seq, optimizer_ntm, optimizer_whole, train_data_loader, valid_data_loader, bow_dictionary, train_bow_loader, valid_bow_loader, opt)

            # mlp_model, criterion, optimizer
            mlp_model = Mlp(ntm_model, vocab_dict, 768, 256)
            # criterion = torch.nn.BCELoss()
            weights = torch.Tensor([1, 150])
            if torch.cuda.is_available():
                mlp_model = mlp_model.cuda()
                weights = weights.cuda()
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # , momentum=0.9)
            optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0001)#args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=50, threshold=1e-4, min_lr=1e-6)
            x = ''
            with open(f'./NTMData/trainVaeEmbeddings{x}.json', 'r') as f:
                train_vae_emb_dict = json.load(f)

            with open(f'./NTMData/validVaeEmbeddings{x}.json', 'r') as f:
                valid_vae_emb_dict = json.load(f)

            test_vae_emb_dict = 0
            # with open(f'./data/StackExchange/testVaeEmbeddings{i}.json', 'r') as f:
            #     test_vae_emb_dict = json.load(f)

            

            # pretrain the mlp model with no vae_loss of bert+lstm+vae+att
            print("warm up!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, bert_dict=text_emb_dict, train_vae_dict=train_vae_emb_dict, valid_vae_dict=valid_vae_emb_dict, test_vae_dict=test_vae_emb_dict, bow_dictionary=invert_dict(bow_dictionary), joint_train_flag=True)
            valid_dataset = ScratchDataset(data_split='Valid', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, bert_dict=text_emb_dict, train_vae_dict=train_vae_emb_dict, valid_vae_dict=valid_vae_emb_dict, test_vae_dict=test_vae_emb_dict, bow_dictionary=invert_dict(bow_dictionary), joint_train_flag=True)

            train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=8)
            valid_dataloader = data.DataLoader(valid_dataset, batch_size=128, collate_fn=my_collate, num_workers=8)
            
            train_mlp(mlp_model, train_dataset, valid_dataset, train_dataloader, valid_dataloader, optimizer, weights, scheduler, joint_train_flag=False, epoch_num=20)            #20

            
            # train the mlp-vae model with the joint loss function
            print('joint train the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # criterion = torch.nn.BCELoss()
            weights = torch.Tensor([1, 150])
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # , momentum=0.9)
            optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.000001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=50, threshold=1e-4, min_lr=1e-6)


            train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, bert_dict=text_emb_dict, train_vae_dict=train_vae_emb_dict, valid_vae_dict=valid_vae_emb_dict, test_vae_dict=test_vae_emb_dict, bow_dictionary=invert_dict(bow_dictionary), joint_train_flag=True)
            valid_dataset = ScratchDataset(data_split='Valid', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, bert_dict=text_emb_dict, train_vae_dict=train_vae_emb_dict, valid_vae_dict=valid_vae_emb_dict, test_vae_dict=test_vae_emb_dict, bow_dictionary=invert_dict(bow_dictionary), joint_train_flag=True)

            train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=8)
            valid_dataloader = data.DataLoader(valid_dataset, batch_size=128, collate_fn=my_collate, num_workers=8)
            
            best_epoch = train_mlp(mlp_model, train_dataset, valid_dataset, train_dataloader, valid_dataloader, optimizer, weights, scheduler, joint_train_flag=True, epoch_num=100)   #100
            
            # test the model 
            #print("test the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, bert_dict=text_emb_dict, train_vae_dict=train_vae_emb_dict, valid_vae_dict=valid_vae_emb_dict, test_vae_dict=test_vae_emb_dict, bow_dictionary=invert_dict(bow_dictionary), joint_train_flag=True)
            
            best_model = Mlp(ntm_model, vocab_dict, 768, 256)
            best_model.load_state_dict(torch.load('./ModelRecords'+f'/model_{best_epoch}.pt'))
            if torch.cuda.is_available():
                best_model = best_model.cuda()
            
            best_model.eval()
            fr = open('./Records/test.dat', 'r')
            fw = open('./Records/test2.dat', 'w')
            lines = fr.readlines()
            lines = [line.strip() for line in lines if line[0] != '#']
            preF = open('./Records/pre.txt', "a")
            last_user = lines[0][6:]
            last_user = last_user.split(' ')[0]
            print('# query 0', file=fw)
            with torch.no_grad():
                x = 0
                for i in tqdm(range(len(test_dataset))):
                    # if x == 10:
                    #     break
                    # x += 1
                    line = lines[i]
                    test_user_feature, test_hashtag_feature, test_bow_hashtag_feature, test_label = test_dataset[i]
                    test_user_feature = test_user_feature.cuda()
                    test_hashtag_feature = test_hashtag_feature.cuda()
                    test_bow_hashtag_feature = test_bow_hashtag_feature.cuda()
                    test_label = test_label.cuda()
                    user = line[6:]
                    user = user.split(' ')[0]
                    if (user == last_user):
                        pass
                    else:
                        print('# query ' + user, file=fw)
                        last_user = user

                    #try:
                    weight, topic_words, pred_label, recon_batch, data_bow, mu, logvar = best_model(True, test_user_feature.unsqueeze(0), torch.tensor([len(test_user_feature)]), test_hashtag_feature.unsqueeze(0), torch.tensor([len(test_hashtag_feature)]), test_bow_hashtag_feature.unsqueeze(0))
                    weight = weight.cpu().detach().numpy().tolist()[0] # list of length 30，each item is to  attention weights of a sentence
                    
                    # type of topic words is a list with length of "n_top_words"
                    # topic_words_str = (str(topic_words.tolist()))

                    print(line, file=fw)

                    # print(pred_label)
                    # print(test_label)

                    pred_label = pred_label.cpu().detach().numpy().tolist()[0]
                    preF.write(f"{pred_label}\n")
                # after_train = criterion(pred_label, test_label)
                # print("test loss after train", after_train.item())

            preF.close()
                  

            training_time = time_since(start_time)

            logging.info('Time for training: %.1f' % training_time)
            
    # except Exception as e:
    #     logging.exception("message")
    return


main(opt)


#if __name__ == "__main__":