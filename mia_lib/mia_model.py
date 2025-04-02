import torch
import torch.nn as nn


"""
This model is referred to paper:

"Machine Learning with Membership Privacy using Adversarial Regularization"

""" 

class RMSNorm(nn.Module):
    def __init__(self, dim=128):
        super(RMSNorm, self).__init__()
        self.dim = dim

    def forward(self, x):

        return x / torch.sqrt((x**2).mean(dim=1, keepdim=True)+1e-6)

##### Version 1 #####

# class Adversary(nn.Module):   #  black-box setting
#     def __init__(self, input_dim=128, attacker_type='default'):
#         super(Adversary, self).__init__()
#         self.input_dim = input_dim
        
#         if attacker_type=='default':
#             self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,512),
#                                      RMSNorm(512),
#                                      nn.Tanh(),
#                                      nn.Linear(512,256),
#                                      RMSNorm(256),
#                                      nn.Tanh(),
#                                      nn.Linear(256,128),
#                                      RMSNorm(128),
#                                      nn.Tanh(),
#                                      nn.Linear(128,1))

#         # init weight
#         for key in self.state_dict():
#             if key.split('.')[-1] == 'weight':
#                 nn.init.normal_(self.state_dict()[key], std=0.01)

#             elif key.split('.')[-1] == 'bias':
#                 self.state_dict()[key][...] = 0

#     def forward(self,x):

#         out = self.pred_fc(x) # B C
#         out = torch.sigmoid(out)
#         return out

#     def init_weights(self,m):
#         if isinstance(m,nn.Linear):
#             m.weight.data.normal_(0,0.01)
#             if m.bias.data is not None:
#                 m.bias.data.fill_(0)




class Adversary(nn.Module):   #  black-box setting
    def __init__(self, input_dim=128, attacker_type='both'):
        super(Adversary, self).__init__()
        self.input_dim = input_dim
        self.attacker_type = attacker_type
        
        if self.attacker_type == "both":
            ## 
            self.image_stream = nn.Sequential(nn.Linear(self.input_dim, 256),
                                     nn.Tanh())

            self.text_stream = nn.Sequential(nn.Linear(self.input_dim, 256),
                                     nn.Tanh())

            self.fusion_stream = nn.Sequential(nn.Linear(512, 256),
                                                nn.Tanh(),
                                                nn.Linear(256,128),
                                                nn.Tanh(),
                                                nn.Linear(128,1))
        if self.attacker_type == "image":
            
            self.image_stream = nn.Sequential(nn.Linear(self.input_dim, 256),
                                     nn.Tanh())

            self.fusion_stream = nn.Sequential(nn.Linear(256, 256),
                                                nn.Tanh(),
                                                nn.Linear(256,128),
                                                nn.Tanh(),
                                                nn.Linear(128,1))

        if self.attacker_type == "text":
            
            self.text_stream = nn.Sequential(nn.Linear(self.input_dim, 256),
                                     nn.Tanh())

            self.fusion_stream = nn.Sequential(nn.Linear(256, 256),
                                                nn.Tanh(),
                                                nn.Linear(256,128),
                                                nn.Tanh(),
                                                nn.Linear(128,1))


        # init weight
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.xavier_uniform_(self.state_dict()[key])

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self,x):
        image, text = torch.split(x, self.input_dim, dim=1)

        if self.attacker_type == "both":
            img_feature = self.image_stream(image)
            txt_feature = self.text_stream(text)
            out = self.fusion_stream(torch.cat([img_feature, txt_feature], dim=1)) # B C
        elif self.attacker_type == "image":
            img_feature = self.image_stream(image)
            out = self.fusion_stream(img_feature) # B C
        else:
            text_feature = self.text_stream(text)
            out = self.fusion_stream(text_feature) # B C

        out = torch.sigmoid(out)

        return out

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            m.weight.data.xavier_uniform_()
            if m.bias.data is not None:
                m.bias.data.fill_(0)