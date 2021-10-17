import torch
import numpy as np
from tqdm import tqdm 
import nltk


def removeHashKey(text):
    text = text.replace('#', '$$$')
    return text


class ScratchDataset(torch.utils.data.Dataset):
    """
    Return (all tensors of user,  all tensors of hashtag, label)
    """

    def __init__(
            self,
            data_split,
            user_list,
            train_file,
            valid_file,
            test_file,
            bert_dict,  # you need to implement load dict of tensors by yourself
            train_vae_dict,
            valid_vae_dict,
            test_vae_dict,
            bow_dictionary,
            joint_train_flag,
            neg_sampling=10,
    ):
        """
        user_list: users occurs in both train, valid and test (which we works on)
        data_file: format of 'twitter_text    user     hashtag1     hashtag2     ...'
        data_split: train/val/test
        """
        self.data_split = data_split
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.neg_sampling = neg_sampling
        self.tag_embed_cap = 5
        self.bert_dict = bert_dict
        self.bow_dictionary = bow_dictionary
        self.joint_train_flag = joint_train_flag
        if (not self.joint_train_flag):
            self.train_vae_dict = train_vae_dict
            self.valid_vae_dict = valid_vae_dict
            self.test_vae_dict = test_vae_dict
        else:
            self.train_vae_dict = bert_dict
            self.valid_vae_dict = bert_dict
            self.test_vae_dict = bert_dict
        self.user_list = user_list
        self.hashtag_split = {}

        self.train_hashtag_list = set()
        self.train_hashtag_per_user = {}
        self.train_text_per_user = {}
        self.train_text_per_hashtag = {}

        self.valid_hashtag_list = set()
        self.valid_hashtag_per_user = {}
        self.valid_text_per_user = {}
        self.valid_text_per_hashtag = {}

        self.test_hashtag_list = set()
        self.test_hashtag_per_user = {}
        self.test_text_per_user = {}
        self.test_text_per_hashtag = {}

        self.user_hashtag = []
        self.label = []

        self.vae_hashtag_dict = {}

        self.process_data_file()
        self.create_dataset()
        # self.getvaefile()
        # self.writevaefile()
    
    def writevaefile(self):
        # write vae file 
        i = 0
        for hashtag in self.vae_hashtag_dict:
            if hashtag == ';':
                continue
            print(i)
            i += 1
            # 20 sentence_list in 1 vae_hashtag_dict[hashtag]   
            src_file = open('../TopicData/test_src.txt', 'a')
            trg_file = open('../TopicData/test_trg.txt', 'a')
            for sentence_list in self.vae_hashtag_dict[hashtag]:
                # print('*'*150)
                # print(type(sentence_list))
                # print(len(sentence_list))
                # print(sentence_list)
                src_token_list = []

                
                # English Tokenize
                # 15 sentence in 1 sentence_list
                for sentence in sentence_list:
                    tokens = nltk.word_tokenize(sentence)
                    # tokens =  jieba.cut(sentence, cut_all=False)
                    src_token_list += tokens

                for token in src_token_list:
                    src_file.write(token+' ')
                src_file.write(hashtag+'\n')
                trg_file.write(hashtag+'\n')
            src_file.close()
            trg_file.close()
    
    def getvaefile(self):
        hashtag_list = []
        for idx, item in enumerate(self.user_hashtag):
            user, hashtag = self.user_hashtag[idx]
            hashtag_list.append(hashtag)
        hashtag_set = set(hashtag_list)
        print(len(hashtag_set))
        for hashtag in tqdm(hashtag_set):
            hashtag_feature = []
        
            spe_tag_posts = []

            # hashtag modeling(train embedding+test others' embedding)

            for sub_hashtag in self.hashtag_split[hashtag]:
                if sub_hashtag in self.train_text_per_hashtag:
                    for text in self.train_text_per_hashtag[sub_hashtag]:
                        spe_tag_posts.append(text)

            if self.data_split == 'Valid':
                texts = []
                for sub_hashtag in self.hashtag_split[hashtag]:
                    texts += self.valid_text_per_hashtag[sub_hashtag]
                for text in list(set(texts) - set(self.valid_text_per_user[user])):
                    spe_tag_posts.append(text)

            if self.data_split == 'Test':
                texts = []
                for sub_hashtag in self.hashtag_split[hashtag]:
                    texts += self.test_text_per_hashtag[sub_hashtag]
                for text in list(set(texts) - set(self.test_text_per_user[user])):
                    spe_tag_posts.append(text)
        
            posts_samples = []
            num = len(spe_tag_posts)
            if num == 0:
                continue
            # print(num)
            for i in range(20):
                one_posts_sample = []
                for j in range(15):
                    k = np.random.randint(num)
                    one_posts_sample.append(spe_tag_posts[k])

                posts_samples.append(one_posts_sample)
            # print(len(posts_samples))
            try:
                a = self.vae_hashtag_dict[hashtag]
            except:
                self.vae_hashtag_dict[hashtag] = posts_samples
    
    def __getitem__(self, idx):
        user, hashtag = self.user_hashtag[idx]
        user_feature, hashtag_feature, bow_hashtag_feature, hashtag_sentences = [], [], [], []
        # user modeling(always train embedding)
        for text in self.train_text_per_user[user]:
            user_feature.append(self.get_feature(self.bert_dict, text))
        # hashtag modeling(train embedding+test others' embedding)

        bow_sentences = []
        for sub_hashtag in self.hashtag_split[hashtag]:
            if sub_hashtag in self.train_text_per_hashtag:
                for text in self.train_text_per_hashtag[sub_hashtag]:
                    bow_sentences.append(text)
                for text in self.train_text_per_hashtag[sub_hashtag]:
                    hashtag_feature.append(self.get_vae_feature(self.train_vae_dict, hashtag, text))
                    hashtag_sentences.append(text)


        # # enrich hashtag embedding contents by user history
        # for text in self.train_text_per_user[user]:
        #     hashtag_feature.append(self.get_feature(self.dict, text))

        if self.data_split == 'Train':
            if len(hashtag_feature) == 0:
                if (not self.joint_train_flag):
                    hashtag_feature.append([0.] * 100)
                else:
                    hashtag_feature.append([0.]*768)

        if self.data_split == 'Valid':
            texts = []
            for sub_hashtag in self.hashtag_split[hashtag]:
                texts += self.valid_text_per_hashtag[sub_hashtag]
            for text in list(set(texts) - set(self.valid_text_per_user[user])):
                bow_sentences.append(text)
            for text in list(set(texts) - set(self.valid_text_per_user[user])):
                hashtag_feature.append(self.get_vae_feature(self.valid_vae_dict, hashtag, text))

            if len(hashtag_feature) == 0:
                if (not self.joint_train_flag):
                    hashtag_feature.append([0.] * 100)
                else:
                    hashtag_feature.append([0.]*768)

        if self.data_split == 'Test':
            texts = []
            for sub_hashtag in self.hashtag_split[hashtag]:
                texts += self.test_text_per_hashtag[sub_hashtag]
            for text in list(set(texts) - set(self.test_text_per_user[user])):
                bow_sentences.append(text)
            for text in list(set(texts) - set(self.test_text_per_user[user])):
                hashtag_feature.append(self.get_vae_feature(self.test_vae_dict, hashtag, text))
                hashtag_sentences.append(text)


            if len(hashtag_feature) == 0:
                if self.joint_train_flag:
                    hashtag_feature.append([0.]*10000)
                elif (not self.joint_train_flag):
                    hashtag_feature.append([0.] * 100)
                else:
                    hashtag_feature.append([0.]*768)

        temp = []
        num = len(hashtag_feature)

        if self.joint_train_flag:
            bow_hashtag_feature = [0.]*10000
            num = len(bow_sentences)
            bow_tokens = []
            for i in range(num):
                # j = np.random.randint(num)
                temp.append(bow_sentences[i])
            for sentence in temp:
                tokens = nltk.word_tokenize(sentence)
                bow_tokens += tokens
            # convert bow_tokens to bow_dictionary
            for token in bow_tokens:
                try:
                    bow_hashtag_feature[self.bow_dictionary[token]] = 1.0
                except:
                    pass

        num = len(hashtag_feature)
    

        temp_feature = []

        for i in range(30):
            j = np.random.randint(num)
            temp_feature.append(hashtag_feature[j])

        hashtag_feature = temp_feature

        user_feature = torch.FloatTensor(user_feature)
        hashtag_feature = torch.FloatTensor(hashtag_feature)
        bow_hashtag_feature = torch.FloatTensor(bow_hashtag_feature)

        return user_feature, hashtag_feature, bow_hashtag_feature, torch.FloatTensor([self.label[idx]])


    def get_feature(self, dict, key):
        try:
            return dict[key]
        except:
            return [0.]*768

    def get_vae_feature(self, dict, tag, text):
        if (not self.joint_train_flag):
            try:
                return dict[tag]
            except:
                return [0.]*100
        else:
            try:
                return self.bert_dict[text]
            except:
                return [0.]*768
        

    def __len__(self):
        return len(self.label)

    # cal user modeling and hashtag modeling
    def process_data_file(self):
        with open('./Data/hashtag_fake_split.csv') as f:
            for line in f:
                l = line.strip('\n').strip('\t').split('\t')
                self.hashtag_split[l[0]] = l[1:]
        f.close()

        trainF = open(self.train_file, encoding='utf-8')
        for line in trainF:
            l = line.strip('\n').split('\t')
            text, user, hashtags = l[0], l[1], l[2:]
            self.train_text_per_user.setdefault(user, [])
            self.train_text_per_user[user].append(text)
            self.train_hashtag_per_user.setdefault(user, set())
            for hashtag in hashtags:
                if len(hashtag) == 0:
                    continue
                self.train_hashtag_list.add(hashtag)
                self.train_hashtag_per_user[user].add(hashtag)
                for sub_hashtag in self.hashtag_split[hashtag]:
                    self.train_text_per_hashtag.setdefault(sub_hashtag, [])
                    self.train_text_per_hashtag[sub_hashtag].append(text)
        trainF.close()

        if self.data_split == 'Valid':
            validF = open(self.valid_file, encoding='utf-8')
            for line in validF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.valid_text_per_user.setdefault(user, [])
                self.valid_text_per_user[user].append(text)
                self.valid_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    if len(hashtag) == 0:
                        continue
                    self.valid_hashtag_list.add(hashtag)
                    self.valid_hashtag_per_user[user].add(hashtag)
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        self.valid_text_per_hashtag.setdefault(sub_hashtag, [])
                        self.valid_text_per_hashtag[sub_hashtag].append(text)
            validF.close()

        if self.data_split == 'Test':
            testF = open(self.test_file, encoding='utf-8')
            for line in testF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.test_text_per_user.setdefault(user, [])
                self.test_text_per_user[user].append(text)
                self.test_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    if len(hashtag) == 0:
                        continue
                    self.test_hashtag_list.add(hashtag)
                    self.test_hashtag_per_user[user].add(hashtag)
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        self.test_text_per_hashtag.setdefault(sub_hashtag, [])
                        self.test_text_per_hashtag[sub_hashtag].append(text)
            testF.close()

    def create_dataset(self):
        """
        Do positive and negative sampling here
        """
        if self.data_split == 'Train':
            for user in self.user_list:
                pos_hashtag = self.train_hashtag_per_user[user]
                neg_hashtag = list(set(self.train_hashtag_list) - set(self.train_hashtag_per_user[user]))
                num = len(neg_hashtag)
                for hashtag in pos_hashtag:

                    # remove hashtag with only x embedding content
                    num_tag_embeds = []
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        if sub_hashtag in self.train_text_per_hashtag:
                            for text in self.train_text_per_hashtag[sub_hashtag]:
                                num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                    
                    if len(num_tag_embeds) >= self.tag_embed_cap:
                        self.user_hashtag.append((user, hashtag))
                        self.label.append(1)
                    for i in range(self.neg_sampling):
                        j = np.random.randint(num)
                        hashtag = neg_hashtag[j]

                        # remove hashtag with only x embedding content
                        num_tag_embeds = []
                        for sub_hashtag in self.hashtag_split[hashtag]:
                            if sub_hashtag in self.train_text_per_hashtag:
                                for text in self.train_text_per_hashtag[sub_hashtag]:
                                    num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                        
                        if len(num_tag_embeds) >= self.tag_embed_cap:
                            self.user_hashtag.append((user, hashtag))
                            self.label.append(0)

        if self.data_split == 'Valid':
            for user in self.user_list:
                pos_hashtag = list(set(self.valid_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                neg_hashtag = list(set(self.valid_hashtag_list) - set(self.valid_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                num = len(neg_hashtag)
                for hashtag in pos_hashtag:

                    # remove hashtag with only x embedding content
                    num_tag_embeds = []
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        if sub_hashtag in self.train_text_per_hashtag:
                            for text in self.train_text_per_hashtag[sub_hashtag]:
                                num_tag_embeds.append(self.get_feature(self.bert_dict, text))

                    texts = []
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        texts += self.valid_text_per_hashtag[sub_hashtag]
                    for text in list(set(texts) - set(self.valid_text_per_user[user])):
                        num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                    
                    if len(num_tag_embeds) >= self.tag_embed_cap:
                        self.user_hashtag.append((user, hashtag))
                        self.label.append(1)
                    for i in range(30):
                        j = np.random.randint(num)
                        hashtag = neg_hashtag[j]

                        # remove hashtag with only x embedding content
                        num_tag_embeds = []
                        for sub_hashtag in self.hashtag_split[hashtag]:
                            if sub_hashtag in self.train_text_per_hashtag:
                                for text in self.train_text_per_hashtag[sub_hashtag]:
                                    num_tag_embeds.append(self.get_feature(self.bert_dict, text))

                        texts = []
                        for sub_hashtag in self.hashtag_split[hashtag]:
                            texts += self.valid_text_per_hashtag[sub_hashtag]
                        for text in list(set(texts) - set(self.valid_text_per_user[user])):
                            num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                        
                        if len(num_tag_embeds) >= self.tag_embed_cap:
                            self.user_hashtag.append((user, hashtag))
                            self.label.append(0)

        if self.data_split == 'Test':
            labelF = open('./Records/test.dat', "a")
            for index, user in enumerate(self.user_list):
                labelF.write(f"# query {index} {user}\n")
                pos_hashtag = list(set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                neg_hashtag = list(set(self.test_hashtag_list) - set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                for hashtag in pos_hashtag:
                    # remove hashtag with only 1 single embedding content
                    num_tag_embeds = []
                    embed_texts = ''
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        if sub_hashtag in self.train_text_per_hashtag:
                            for text in self.train_text_per_hashtag[sub_hashtag]:
                                num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                                embed_texts += ('+++++'+text)

                    texts = []
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        texts += self.test_text_per_hashtag[sub_hashtag]
                    for text in list(set(texts) - set(self.test_text_per_user[user])):
                        num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                        embed_texts += ('+++++'+text)
                    
                    embed_texts = removeHashKey(embed_texts)
                    
                    if len(num_tag_embeds) >= self.tag_embed_cap:
                        self.user_hashtag.append((user, hashtag))
                        self.label.append(1)
                        # if you need to analyze the embedded texts, you can run this line instead of "labelF.write(f"{1} qid:{index}\n")"
                        # labelF.write(f"{1} qid:{index} {hashtag} {embed_texts}\n")
                        labelF.write(f"{1} qid:{index}\n")
                
                num = len(neg_hashtag)
                for i in range(len(pos_hashtag) * 100):
                    j = np.random.randint(num)
                    hashtag = neg_hashtag[j]

                    # remove hashtag with only 1 single embedding content
                    num_tag_embeds = []
                    embed_texts = ''
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        if sub_hashtag in self.train_text_per_hashtag:
                            for text in self.train_text_per_hashtag[sub_hashtag]:
                                num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                                embed_texts += ('+++++'+text)

                    texts = []
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        texts += self.test_text_per_hashtag[sub_hashtag]
                    for text in list(set(texts) - set(self.test_text_per_user[user])):
                        num_tag_embeds.append(self.get_feature(self.bert_dict, text))
                        embed_texts += ('+++++'+text)
                    if len(num_tag_embeds) == 0:
                        num_tag_embeds.append([-1.0])
                    
                    embed_texts = removeHashKey(embed_texts)
                    
                    if len(num_tag_embeds) >= self.tag_embed_cap:
                        self.user_hashtag.append((user, hashtag))
                        self.label.append(0)
                        # if you need to analyze the embedded texts, you can run this line instead of "labelF.write(f"{0} qid:{index}\n")"
                        # labelF.write(f"{0} qid:{index} {hashtag} {embed_texts}\n")
                        labelF.write(f"{0} qid:{index}\n")
               
            labelF.close()

    def load_tensor_dict(self):
        raise NotImplementedError


def my_collate(batch):
    user_features, hashtag_features, bow_hashtag_features, labels = [], [], [], []
    user_len, hashtag_len = [], []
    users, items = [], []
    batch_size = len(batch)
    for user_feature, hashtag_feature, bow_hashtag_feature, label in batch:
        # text features
        user_features.append(user_feature)
        hashtag_features.append(hashtag_feature)
        bow_hashtag_features.append(bow_hashtag_feature)
        labels.append(label)
        user_len.append(user_feature.shape[0])
        hashtag_len.append(hashtag_feature.shape[0])

    max_user_len, max_hashtag_len = max(user_len), max(hashtag_len)
    for i in range(batch_size):
        user_feature, hashtag_feature = user_features[i], hashtag_features[i]
        user_features[i] = torch.cat((user_feature, torch.zeros(max_user_len-len(user_feature), 768)), dim=0)
        hashtag_features[i] = torch.cat((hashtag_feature, torch.zeros(max_hashtag_len-len(hashtag_feature), 768)), dim=0)
    user_features, hashtag_features, bow_hashtag_features = torch.stack(user_features), torch.stack(hashtag_features), torch.stack(bow_hashtag_features)
    user_len, hashtag_len, labels = torch.tensor(user_len), torch.tensor(hashtag_len), torch.tensor(labels)
    return user_features, user_len, hashtag_features, hashtag_len, bow_hashtag_features, labels
