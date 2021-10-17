import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Attention(nn.Module):
    def __init__(self, rnn_size, hidden_size):
        super(VAE_Attention, self).__init__()
        self.rnn_size =rnn_size
        self.att_hid_size = hidden_size
        #self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        #self.z2att = nn.Linear(100, self.att_hid_size)

    def forward(self, h, att_feats, p_att_feats, z_feature, att_masks=None):
    # h:[batch, hidden_size] 
    # att_feats:[batch,att_size, hidden_size]
    # p_att_feats:[batch,att_size, hidden_size]
        if att_masks is not None:
            length = att_feats.size(1)
            att_masks = att_masks[:, :length]

        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        # print(att.size()) # torch.Size([128, 30, 768])
        att_h = h#self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        #att_z = self.z2att(z_feature)
        #att_z = att_z.unsqueeze(1).expand_as(att)
        dot = att + att_h #+ att_z # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return weight, att_res
    