import torch
from Modules.vae_attention import VAE_Attention
import torch.nn.functional as F


class Mlp(torch.nn.Module):
    def __init__(self, vae, vocab_dict, input_size, hidden_size):
        super(Mlp, self).__init__()
        self.vae = vae
        self.vocab_dict = vocab_dict
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.relu = torch.nn.ReLU()
        self.Vae_Attention = VAE_Attention(self.input_size, self.input_size) # att_hidden=512

        self.fc1 = torch.nn.Linear(self.input_size * 3, self.hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(num_features=self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.fc3 = torch.nn.Linear(self.input_size * 2, self.input_size) # from user embeds768*2 to attention hashtag embeds768
        self.fc4 = torch.nn.Linear(self.input_size+50, self.input_size)
        self.fc5 = torch.nn.Linear(self.input_size+50, self.input_size)
        self.lstm1 = torch.nn.LSTM(768, 768, num_layers=2, bidirectional=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, joint_train_flag, user_features, user_lens, hashtag_features, hashtag_lens, bow_hashtag_features):
        user_embeds = self.user_modeling(user_features, user_lens)
        weight, word_weight, hashtag_embeds, recon_outputs, data_bow_outputs, mu_outputs, logvar_outputs = self.hashtag_modeling(user_features, user_lens, hashtag_features, hashtag_lens, joint_train_flag, bow_hashtag_features, user_embeds=user_embeds)
        x = torch.cat((user_embeds, hashtag_embeds), dim=1)
        x = self.relu(self.bn1(self.fc1(x)))
        output = self.fc2(x)

        output = self.sigmoid(output)
        return weight, word_weight, output, recon_outputs, data_bow_outputs, mu_outputs, logvar_outputs
    
    def get_vae_model(self):
        return self.vae

    def user_modeling(self, user_features, user_lens):
        #user_features = self.fc5(user_features)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(user_features, user_lens.cpu(), batch_first=True, enforce_sorted=False) # torch.Size([y, 768]), y是序列长度总和(128+128+100+96+..+1)
        packed_output, (h, c) = self.lstm1(inputs)  #packed_output: torch.Size([y, 768]), y是序列长度总和(128+128+100+96+..+1) 
        outputs = torch.cat((h[-2], h[-1]), dim=1)
        return outputs # torch.Size([128, 768*2])

    def hashtag_modeling(self, user_features, user_lens, hashtag_features, hashtag_lens, joint_train_flag, bow_hashtag_features, user_embeds=None):
        user_query = self.fc3(user_embeds) # torch.Size([128, 1, 768*2]) -> torch.Size([128, 1, 768])
        att_key = hashtag_features#self.fc7(hashtag_features) # torch.Size([128, 30, 768])
        att_value = hashtag_features#self.fc8(hashtag_features) # torch.Size([128, 30, 768])
        masks = torch.where(hashtag_features[:, :, 0] != 0, torch.Tensor([1.]).cuda(), torch.Tensor([0.]).cuda())
        data_bow_outputs = []
        bow_hashtag_feature = bow_hashtag_features

        data_bow_outputs.append(bow_hashtag_feature) 
        data_bow = torch.stack(data_bow_outputs)
        data_bow_norm = F.normalize(data_bow) 
        _, z_feature, recon_batch, mu, logvar = self.vae(data_bow_norm)

        # print topic words of hashtag sentence
        #topic_words = self.vae.print_case_topic(self.vocab_dict, z_feature, n_top_words=30)
        #word_weight = self.vae.print_word_weight(self.vocab_dict, z_feature)

        z_features = []
        for i in range(30):
            z_features.append(z_feature)
        z_features = torch.stack(z_features)
        z_features = z_features.transpose(0,1).contiguous()
        
        att_key = torch.cat((att_key, z_features), 2)
        att_value = torch.cat((att_value, z_features), 2)

        att_key = self.fc4(att_key)
        att_value = self.fc5(att_value)

        weight, hashtag_embeds = self.Vae_Attention(user_query, att_key, att_value, z_features, masks)

        return weight, _, hashtag_embeds, recon_batch, data_bow, mu, logvar

