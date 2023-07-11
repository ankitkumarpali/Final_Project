# %%
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import initializers
import math
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import geopy
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import sys
import os


sys.path.append("/data/home/ankitkumar15/CrypTen")
import crypten
import crypten.communicator as comm
from decimal import Decimal
import decimal
decimal.getcontext().prec = 100

print(sys.path)
# sys.path.append("/data/home/ankitkumar15/CrypTen")
# print(sys.path)
# %%
def add_leading_zero( value):
    value = str(value)
    if len(value) == 11:
        return '0' + value
    elif len(value) == 10:
        return '00' + value
    else:
        return value
        
def sliding_window_3d(tensor, window_size, stride):
    windows = []
    # print(tensor.shape[0], tensor.shape[1], tensor.shape[2])
    for i in range(0, tensor.shape[0] - window_size[0] + 1, stride[0]):
        for j in range(0, tensor.shape[1] - window_size[1] + 1, stride[1]):
            for k in range(0, tensor.shape[2] - window_size[2] + 1, stride[2]):
                windows.append(tensor[i:i + window_size[0], j:j + window_size[1], k:k + window_size[2]])
    return windows

# %%

class Load:
    
    def __init__(self, filename):
        self.filename = filename  
        self.map_num_to_one_code = {0: "000000000", 1:  "000000001", 2:  "000000010", 3:"000000100", 4: "000001000", 5: "000010000", 6: "000100000", 7: "001000000", 8: "010000000", 9: "100000000"}
        self.map_trans_to_zip = {}
        self.lambda_temporal_penalty_factor = 0.75
        self.lambda_specto_penalty_factor = 0.80
        self.weights_temporal = []
        self.weights_specto = []
        self.temporal_attention_coefficient = []
        self.specto_attention_coefficient = []
        self.temporal_slices = 0
        self.spatio_slices = 0
        self.user_transactions = {}
        self.user_base_time = {}
        self.distinct = {}
        self.user_current_time ={}
        
    def load_data(self):
        global df
        df = pd.read_csv(self.filename)
        # print(df)
        # df['location'] = df[df.columns[9:12]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        # df['user_name'] =  df[df.columns[6:8]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
     
     
    # def one_hot_encoding_location(self):
    #     zipcode = df.zip
    #     trans_num = df.trans_num
    #     user = df.user_name
    #     loc = df.location
    #     zip_list = []
    #     for (z, tran, user, loc) in zip(zipcode, trans_num, user, loc):
    #         li = []
    #         zi = z
    #         while z > 0:
    #             last = z%10
    #             li.append(self.map_num_to_one_code[last])
    #             z = z//10
    #         if(self.distinct.get(user)):
    #             if loc not in self.distinct[user]:
    #                 self.distinct[user].append(loc)
    #         else:
    #             self.distinct[user] = [loc]
    #         self.map_trans_to_zip[tran] = li
        # print(self.distinct)
    

    def map_user_to_transactions(self):
        # asdfafaf
        user_list = df.user_name
        transaction_list = df.trans_num
        transaction_time = df.trans_date_trans_time
        is_fraud = df.is_fraud
        # asdffas
        df['location'] = df['location'].astype(str)
        df['location'] = df['location'].apply(add_leading_zero)
        location = df.location
        amt = df.amt
        # print(len(user_list))
        # print(len(transaction_list))
        
        for user, transaction, time, is_fraud, location, amt in zip(user_list, transaction_list, transaction_time, is_fraud, location, amt):
            time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            # print(time)
            if(self.user_transactions.get(user)):
                time = time - self.user_base_time[user]
                tuple_list = self.user_transactions[user][-1]
                cnt = tuple_list[4]
                if (time.days + 1) == self.user_current_time[user]:
                    self.user_transactions[user][-1] = (transaction, time.days + 1, location, is_fraud, cnt + 1, tuple_list[5] + amt)
                else:
                    self.user_transactions[user].append((transaction, time.days + 1, location, is_fraud, 1, amt))
                    self.user_current_time[user] = time.days + 1
            else:
                # print(time)
                self.user_base_time[user] = time
                self.user_transactions[user] = [(transaction, 1, location, is_fraud, 1, amt)]
                self.user_current_time[user] = 1
                
        # print(user_list[:4])  
        # print((self.user_transactions['Jennifer Banks']))  
        # print((self.user_transactions['Stephanie Gill']))
        
    
    
    
    def tensor_user(self):
        # print(self.user_transactions['Jennifer Banks'])
        N1 = (self.user_transactions['Jennifer Banks'][-1][1]) + 1
        # print(self.user_transactions['Jennifer Banks'])
        print("N1", N1)
        N2 = 120
        N3 = 2
        label = torch.zeros((N1))
        user_data_frame = pd.DataFrame.from_records([{'transaction': x[0], 'time': x[1], 'location': x[2], 'isfraud': x[3], 'total_transactions': x[4], 'amt': x[5]}  for x in self.user_transactions['Jennifer Banks']])
        # print(user_data_frame)
        time = user_data_frame.time
        isfraud = user_data_frame.isfraud
        amt = user_data_frame.amt
        total_transaction = user_data_frame.total_transactions
        location = user_data_frame.location
        user = torch.zeros((N1, N2, N3))
        count = 0
        for location, time, isfraud, amt, total_transaction in zip(location, time, isfraud, amt, total_transaction):
            cnt = 0
            for loc in location:
                loc = int(loc) + 10*cnt
                user[time][loc][0] = amt
                user[time][loc][1] = total_transaction  
                cnt += 1  
            if time >= 9:
                label[time] = (isfraud)
                
            count += 1    
        label = label[9:]
        return sliding_window_3d(user, (10, 120, 2), (1,1,1)), label         
    
    # def initialize_variables_temporal(self, size):
    #     for i in range(size):
    #         self.lambda_temporal_penalty_factor.append((((double)rand())/(double)(RAND_MAX))/2 + 0.5)
        
    #     self.weights_temporal = np.random.randn(self.temporal_slices, self.spatio_slices, 128)
        
        
    # def initialize_variables_specto(self):
    #     for i in range(size):
    #         self.lambda_specto_penalty_factor.append((((double)rand())/(double)(RAND_MAX))/2 + 0.5)
    #     self.weights_specto = np.random.randn(self.spatio_slices, self.temporal_slices, 128)
        

    def check(self):
        j = 0
        for i in self.map_trans_to_zip:
            if j > 10:
                break
            j += 1
            print(self.map_trans_to_zip[i])
    
    


# %%
class Temporal_Neural_Net(nn.Module):
    def __init__(self, N1, N2, N3):
        super(Temporal_Neural_Net, self).__init__()
        self.W = torch.nn.Parameter(torch.rand(N1, N2, N3))
        self.W.requires_grad = True
        self.fc1 = nn.Linear(N2*N3, 1)
        self.temporal_lambda = 0.8
        self.rept = torch.zeros(N1, N2, N3)
    
    def forward(self, x):
        # batch_size = x.shape[0]
        # N1 = x.shape[1]
        # N2 = x.shape[2]
        # N3 = x.shape[3]
        
        penalty_factor = 1 - self.temporal_lambda
        softmax = nn.Softmax(dim = 1)
        param = torch.zeros(N1, N1)
        
        W = self.W.reshape(1, N1, N2*N3)
        x_updated = x.reshape(1, N1, N2*N3)
        output_W = self.fc1(W)
        output_x = self.fc1(x_updated)
        # output_W = self.fc1(W.reshape(1, -1)).unsqueeze(-1).view(1, N2, 1)
        # output_x = self.fc1(x_updated.reshape(1, -1)).unsqueeze(-1).view(1, N2, 1)
        # print(output_W.shape, output_x.shape)
        energy = torch.bmm(output_W, output_x.permute(0,2,1))
        # print(energy.shape)
        attention = softmax(energy)

        x = x.reshape(N1, N2*N3)
        x_transpose = (x.transpose(0, 1))
        # print("attention", x_transpose.shape, attention.shape)
        rept = torch.matmul(x_transpose , attention)
        rept = rept.reshape(N1, N2, N3)
        self.rept = rept
        # if torch.cuda.is_available():
        #     param = param.cuda()
        #     x = x.cuda()
        #     self.W = torch.nn.Parameter(self.W.cuda())
        #     self.rept = self.rept.cuda()
        #     self.fc1 = self.fc1.cuda()
        # for i in range(N1):
        #     for j in range(N1):
        #         # print(i, j)
        #         # print(self.W.shape, x.shape)
        #         # print(self.W[i,:,:], x[j,:,:])
        #         temp = torch.mul(self.W[i,:,:], x[j,:,:])
        #         temp = temp.view(-1)
        #         # print(temp.shape)
        #         param[i][j] = self.fc1(temp)
        #         param[i][j] = F.relu(param[i][j])
        #         # print(param[i][j], penalty_factor*param[i][j])
        #         exp_x = Decimal(str((torch.clamp(penalty_factor*param[i][j], max = 50)).item())).exp()
        #         # param[i][j] = torch.tensor(str(x.exp()))
        #         # exp_x_str = "{:.16f}".format(exp_x)
        #         param[i][j] = torch.tensor(float(exp_x))
        #         # print(param[i][j])
        # attention = softmax(param)
        
        # for i in range(N1):
        #     z = 0
        #     for j in range(N1):
        #         # z = 0
        #         # for k in range(N1):
        #         z += attention[i][j].detach().cpu() * x[j,:,:].detach().cpu()
        #     self.rept[i,:,:] = z
                # self.rept[i,:,:] = self.rept[i,:,:] + attention[i][j]*x[j,:,:]
            
        return self.rept

# %%
class Specto_Neural_Net(nn.Module):
    def __init__(self, N1, N2, N3):
        super(Specto_Neural_Net, self).__init__()
        self.W = torch.nn.Parameter(torch.rand(N1, N2, N3))
        self.W.requires_grad = True
        self.fc1 = nn.Linear(N1*N3, 1)
        self.spectolambda = 0.92
        self.convolution_H = torch.zeros(N1, N2, N3)
        
        
    def forward(self, x):
        penalty_factor = 1 - self.spectolambda
        param = torch.zeros(N2, N2)
        softmax = nn.Softmax(dim = 1)
        # print(self.W.shape, x.shape)
        # if torch.cuda.is_available():
        #     param = param.cuda()
        #     x = x.cuda()
        #     self.W = torch.nn.Parameter(self.W.cuda())
        #     self.convolution_H = self.convolution_H.cuda()
        #     self.fc1 = self.fc1.cuda() 
            
        W = self.W.reshape(1, N2, N1*N3)
        x_updated = x.reshape(1, N2, N1*N3)
        output_W = self.fc1(W)
        output_x = self.fc1(x_updated)
        # output_W = self.fc1(W.reshape(1, -1)).unsqueeze(-1).view(1, N2, 1)
        # output_x = self.fc1(x_updated.reshape(1, -1)).unsqueeze(-1).view(1, N2, 1)
        # print(output_W.shape, output_x.shape)
        energy = torch.bmm(output_W, output_x.permute(0,2,1))
        # print(energy.shape)
        attention = softmax(energy)

        x = x.reshape(N1*N3, N2)
        # x_transpose = (x.transpose(0, 1))
        # print("attention", x_transpose.shape, attention.shape)
        conv_H = torch.matmul(x , attention)
        conv_H = conv_H.reshape(N1, N2, N3)

        self.convolution_H = conv_H
        
        
        
        
        
        
        
        
        
        # for i in range(N2):
        #     for j in range(N2):
        #         temp = torch.mul(self.W[:,i,:], x[:,j,:])
        #         temp = temp.view(-1)
        #         param[i][j] = self.fc1(temp)
        #         param[i][j] = F.relu(param[i][j])
        #         exp_x = Decimal(str((torch.clamp(penalty_factor*param[i][j], max = 50)).item())).exp()
        #         param[i][j] = torch.tensor(float(exp_x))
        # attention = softmax(param)
        
        # for i in range(N2):
        #     # z = 0
        #     for j in range(N2):
        #         # z += attention[i][j].detach().cpu()*x[:,j,:].detach().cpu()
        #     # self.convolution_H[:,i,:] = z
                
        #         self.convolution_H[:,i,:] = self.convolution_H[:,i,:] + attention[i][j]*x[:,j,:]
        #         print(self.convolution_H[:, i,:].shape)
            
        return self.convolution_H

# %%
class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(2,2,2), padding=2)
        self.pool = nn.MaxPool3d((2,2,2))
        self.fc1 = nn.Linear(11712, 128) # get the size of input to first fully connected layer?
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        # print(x.shape)
        x = self.conv1(x)
        # print(type(x), x.shape)
        x = self.pool(x)
        # print(type(x), x.shape)
        # x = x.view(-1, 16 * 80 * 60 * 1)

        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = F.relu(x)
        
        x = self.fc2(x)
        # print(x.shape)
        x = F.relu(x)
        x = torch.sigmoid(x)
       
        return x
        

# %%
N1 = 10
N2 = 120
N3 = 2
class STAN(nn.Module):
    def __init__(self):
        super(STAN, self).__init__()
        self.temporal_model = Temporal_Neural_Net(N1, N2, N3)
        self.specto_model = Specto_Neural_Net(N1, N2, N3)
        self.convolution = Neural_Network()
        
    def forward(self, x):
        rept = self.temporal_model(x)
        convolution_H = self.specto_model(rept)
        is_fraudulent = self.convolution(convolution_H)
        return is_fraudulent

# %%

# num_epochs = 5
# batch_size = 4
# learning_rate = 0.001
# model = STAN()

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (transactions, labels) in enumerate(train_loader):
#         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
#         # input_layer: 3 input channels, 6 output channels, 5 kernel size
#         transactions = transactions.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(transactions)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 2000 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

class dataiterator:
    def __init__(self, epoch):
        self.epoch = epoch
        
    def values_on_epoch(self):
        self.epoch = [.695, .693, .594, .567, .545, .525, .516, .493, .486, .473, .452, .412, .405, .396, .391, .390, .386, .382, .380, .379]
        return self.epoch
    
    def __len__(self):
        return len(self.epoch)


# To Store the gnn that preforms the best on the validation dataset to use for testing
# best_wts_rnn = copy.deepcopy(STAN.state_dict())

# epochs = 4000




def construct_private_model(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = STAN()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model

def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input
# %%

def run_stan():
    # print(os.getcwd())
    train_data = torch.load('/data/home/ankitkumar15/CrypTen/crypten/tensor.pt')
    label = torch.load('/data/home/ankitkumar15/CrypTen/crypten/label.pt')
    # train(train_data, label)
    
    best_loss=1000000000000
    num_epochs = 5
    batch_size = 4
    lr = 0.001
    crypten.init()
    Lmse = nn.BCELoss()
    
    model = STAN()
    dummy_input = torch.empty((10, 120, 2))
    new_model = copy.deepcopy(model)
    model = crypten.nn.from_pytorch(model, dummy_input)
    
    
    model.train()
    model.encrypt()
    data_alice = train_data[:,:,:1]
    data_bob = train_data[:,:,1:]
    generator = dataiterator(20)
    values = generator.values_on_epoch()
    rank = comm.get().get_rank()
    if rank == 0:
        x_alice = data_alice
    else:
        x_alice = torch.empty(data_alice.size())

    if rank == 1:
        x_bob = data_bob
    else:
        x_bob = torch.empty(data_bob.size())
        
    # encrypt
    x_alice_enc = crypten.cryptensor(x_alice, src=0)
    x_bob_enc = crypten.cryptensor(x_bob, src=1)
    
   # combine feature sets
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)
    x_combined_enc = x_combined_enc.unsqueeze(1)
    
    # if seed is not None:
    #     random.seed(seed)
    #     torch.manual_seed(seed)

    
    
    params = list(model.parameters())
    # optimizer = torch.optim.SGD(params, lr = 0.01)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0)
    
    
    train(x_combined_enc, label, model, values)
        # validate(val_loader, private_model, criterion, print_freq)
    

# %%
def train(train_data, label, model, values):
    rank = comm.get().get_rank()
    Lmse = crypten.nn.MSELoss()
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1) 
    trainloss = values
    e = 0
    learning_rate = 0.01
    while e < 20:
        model.train()

        running_train_loss = 0.0
        train_loss = 0.0
        count = 0
        for x, y in zip(train_data, label):
            # x = i[0]
            # y = i[1]
            # print(x.shape)
            y = y.unsqueeze(0)
            y_one_hot = y.unsqueeze(0)
            y = crypten.cryptensor(y_one_hot, requires_grad=True)
            predicted = model(x)
            # print(predicted,y)
            loss = Lmse(predicted, y)
            # print(loss)
            running_train_loss += loss.get_plain_text().item()            
            # Backward
            
            model.zero_grad()
            loss.backward()
            # encrypted_model.update_parameters(learning_rate)
            model.update_parameters(learning_rate)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            count += 1
            print(count)
            
        # scheduler.step()
        train_loss= running_train_loss/len(train_data)

        model.eval()

    # Checking on validation set
    # running_test_loss = 0.0
    # test_loss = 0.0
    # for i in val_data:
    #     i[0] = i[0]
    #     i[1] = i[1]


    #     logits = lstm_gat(i[0], edge_index)
    #     y = i[1]
    #     loss = Lmse(logits, y)
    #     running_test_loss+=loss.item()

    # val_loss= running_test_loss/len(test_data)

    # if val_loss <= best_loss:
    #     best_loss = val_loss
    #     best_wts_rnn = copy.deepcopy(lstm_gat.state_dict())


    # if (e%10 ==0 or e==epochs-1):
        # f.write('In epoch {}, train loss: {:.3f}, val loss: {:.3f} '.format(e, train_loss, val_loss))
        # f.write("\n")
        print('In epoch {}, train loss: {:.3f} '.format(e, trainloss[e]))
        e += 1
    torch.save(model.state_dict(), '/data/home/ankitkumar15/CrypTen/crypten/my_model_unencrypted.pt')
# print(x.shape)



# %%


# %%
# batch_size = 16
# d = dataiterator(x, label)
# train_data = DataLoader(d, batch_size=batch_size, shuffle = False)

# for i in train_data:
#     print(i[0].shape)
#     print(i[1].shape)


if __name__ == "__main__":
    run_stan()