import torch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os

class model(nn.Module):
    def __init__(self, layers, name=""):
        super(model, self).__init__()
        self.name = ""
        self.layers = layers
        self.features = nn.Sequential(*layers)


    def name(self):
        return self.name

    def model_summery(self):
        return self.features.summary()

    def forward(self, x):
        return self.features(x)

class Oven:
    def __init__(self, input_size=51, optimizer=None, dropout=None, epoch=10, learning_rate=0.001, batch_size=100):
        """
        init the object
        :param input_size:
        :param optimizer: which oprimaizer to run
        :param dropout: if it use dropout, the dropout rate
        :param epoch:
        :param learning_rate:
        :param batch_size:
        """
        self.input_size = input_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_debug = False
        self.debug_level = 10
        self.total_loss = []
        self.total_accuracy = []
        self.Net = None

        self.lr_start = 0.05
        self.lr_end= 0.5
        self.lr_step = 0.05
        if not optimizer:
            self.optimizers_list = {"SGD": optim.SGD, "ADAM": optim.Adam, "RMSprop": optim.RMSprop, "AdaDelta": optim.Adadelta}
        else:
            self.optimizers_list = optimizer

        self.optimizer = None
        self.layers = None

        #for nice output
        self.epoch_i = 0
        #cpu options
        self.dev = "cpu"

    def set_epoch(self,epoch):
        self.epoch = epoch

    def use_gpu(self, use=True):
        if torch.cuda.is_available() and use:
            self.dev = "cuda"
            self.dtype = torch.cuda.FloatTensor
            print("[!]using cuda GPU")
        else:
            self.dev = "cpu"
            print("[!]using CPU")
        if self.Net !=None:
            self.Net.to(self.dev)



    def set_layers(self,layers):
        self.layers = layers

    def step_train(self, train_loader, epoch_i =10):
        """
        train the model
        :param train_loader: dataset
        :param epoch_i: in which epoch am i
        :return:
        """
        self.Net.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.Net.forward(data.to(self.dev))
            loss = F.nll_loss(output, labels.to(self.dev))
            # self.total_loss.append(loss.data)
            loss.backward()
            self.optimizer.step()
            self.until_now =batch_idx * self.batch_size
            self.data_set_len = len(train_loader.dataset)
            if self.print_debug:
                print(f"\tTrain Epoch: {epoch_i} [{self.until_now}/{self.data_set_len}"
                      f" ({round(100. * self.until_now / self.data_set_len, 2)}%)]  loss: {loss}")

    def validate(self, test_loader):
        """
        test the modal
        :param test_loader:
        :return: the accuracy of this epoch
        """
        self.Net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                target = target.to(self.dev)
                data = data.to(self.dev)
                output = self.Net(data)
                try:
                    test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                except:
                    print(data,output, target)

                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).cpu().sum()
                correct += pred.eq(target.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            curr_accuracy = 100. * correct / len(test_loader.dataset)
            self.total_loss.append(test_loss)
            self.total_accuracy.append(curr_accuracy)
            print("\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, correct, len(test_loader.dataset),curr_accuracy))
            return curr_accuracy

    def train_and_vaildate(self,train_loader, test_loader):
        """
        train and validate in each epoch
        :param train_loader:
        :param test_loader:
        :return:
        """
        self.total_loss = []
        self.total_accuracy = []
        for epoch_i in range(self.epoch):
            print(f"[!]Train Epoch: {epoch_i}")
            self.step_train(train_loader, epoch_i)
            curr_accuracy = self.validate(test_loader)
            self.epoch_i = epoch_i
            self.pickle_save(path=f'results/epoch_{self.epoch_i}_{curr_accuracy}',filename=curr_accuracy,info=f'accuracy: {curr_accuracy}')
            # torch.save(self.Net.state_dict(), f"save_State{epoch_i}.tourch")

    def do_train(self,train_loader):
        """
        train in each epoch
        :param train_loader:
        :param test_loader:
        :return:
        """
        for epoch_i in range(self.epoch):
            print(f"[!]Train Epoch: {epoch_i}")
            self.step_train(train_loader, epoch_i)

    def test_step(self, test_input):
        """
        test the modal
        :param test_input:
        :return: the accuracy of this epoch
        """
        self.Net.eval()
        with torch.no_grad():
            return self.Net(test_input).argmax()

    def test (self,test_loader,classes,file_name="y_test"):
        with open(file_name,"w") as output_file:
            for i, (data, labels) in enumerate(test_loader):
                # print(f"{os.path.basename(test_loader.dataset.spects[i][0])},{self.test_step(data)}")
                output_file.write(f"{os.path.basename(test_loader.dataset.spects[i][0])}, {classes[self.test_step(data)]}\n")

    def showGraphs(self,save_path=None):
        """
        show the required graphs
        :return:
        """
        import matplotlib.pyplot as plt
        plt.plot(range(1,self.epoch_i+2), self.total_loss, label=f'Loss - {self.Net.name} ')
        plt.legend(bbox_to_anchor=(1.0, 1.00))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.xticks(plt_learning)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(f"{save_path}/Loss.png")
        plt.close()
        plt.plot(range(1,self.epoch_i+2), self.total_accuracy, label=f'Accuracy - {self.Net.name} ')
        plt.legend(bbox_to_anchor=(1.0, 1.00))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.xticks(plt_learning)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(f"{save_path}/Accuracy.png")

    def best_Values_for_model_without_droupout(self,train_loader, test_loader):
        """
        find the best args for the models
        :param train_loader:
        :param test_loader:
        :param model: the model to run
        :return:
        """
        best_accuracy = 0
        best_lr = 0
        best_optimazier = ""

        for optimazier_name, optimazier_func in self.optimizers_list.items():
            print(f"[!]Checking {optimazier_name}")
            for lr in np.arange(self.lr_start, self.lr_end, self.lr_step):

                self.Net = model(layers=self.layers)
                if self.dev != "cpu":
                    self.Net.cuda()

                self.optimizer = optimazier_func(self.Net.parameters(), lr=lr)

                self.step_train(train_loader, epoch_i=1)

                print(f"[+]test on: lr: {lr} optimaize: {optimazier_name}")
                curr_accuracy = self.validate(test_loader)
                if best_accuracy < curr_accuracy:
                    best_accuracy = curr_accuracy
                    best_lr = lr
                    best_optimazier = optimazier_name
                    print(
                        f"[!!!]found better parrams:  accuracy: {best_accuracy} lr: {lr} optimaize: {best_optimazier} ")
        print(best_accuracy, best_lr, best_optimazier)
        return (best_accuracy, best_lr, best_optimazier)

    def model_summery(self):
        return self.Net.model_summery()

    def pickle_save(self,path='results/',filename='',info=""):
        import os
        os.makedirs(path, exist_ok=True)
        pickle.dump(self, open(f"{path}/{filename}.pkl", "wb"))
        info_file = open(f"{path}/{filename}_info.txt", "w")
        info_file.write("-------------Layers-----------\n")
        info_file.write(str(self.layers)+"\n")
        info_file.write("-------------Epochs-----------\n")
        info_file.write(str(self.epoch_i)+"\n")
        info_file.write("-------------Learning rate-----------\n")
        info_file.write(str(self.learning_rate) + "\n")
        info_file.write("-------------Loss rate-----------\n")
        info_file.write(str(self.total_loss[-1]) + "\n")
        info_file.write("-------------Accuracy rate-----------\n")
        info_file.write(str(self.total_accuracy[-1]) + "\n")
        info_file.write("------------------------------"+"\n")
        info_file.write(info+"\n")
        self.showGraphs(path)




class Chef:
    """
    class that manage the network
    """
    def __init__(self,train_loader, test_loader, input_size=51, optimizer=None, dropout=None, epoch=10, learning_rate=0.001, batch_size=100):
        """
        init the object
        :param input_size:
        :param optimizer: which oprimaizer to run
        :param dropout: if it use dropout, the dropout rate
        :param epoch:
        :param learning_rate:
        :param batch_size:
        """
        #set data information
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_size = input_size
        self.batch_size = batch_size
        self.label_size = 2

        #set train info
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.dropout = dropout
        if not optimizer:
            self.optimizers_list = {"SGD": optim.SGD, "ADAM": optim.Adam, "RMSprop": optim.RMSprop, "AdaDelta": optim.Adadelta}
        else:
            self.optimizers_list = optimizer

        #set print option
        self.print_debug = False
        self.debug_level = 10 #TODO:add print debug

        #set network info
        self.depth_size = 3
        self.features_size_to_test = [52,256,512,1024,2046,52*2,52*4,52*6]
        self.features_size_to_test = [52,256]

        #save result
        self.best_accuracy = 0
        self.best_net = None
        self.dev = "cpu"

    def set_epoch(self,epochs):
        self.epoch = epochs

    def use_gpu(self,use=True):
        if torch.cuda.is_available() and use:
            self.dev = "cuda"
            self.dtype = torch.cuda.FloatTensor

            print("[!]using cuda GPU")
        else:
            self.dev = "cpu"
            print("[!]using CPU")

    def set_network_length(self,length):
        self.depth_size = length

    def set_features_size_list(self,f_list):
        self.features_size_to_test = f_list

    def find_best_arc(self):
        for next_size in self.features_size_to_test:
            first = [nn.Linear(self.input_size,next_size) ,
                               nn.BatchNorm1d(next_size),nn.ReLU(inplace=True)]
            self._recursiv_build_network_to_test(first,next_size)

    def _recursiv_build_network_to_test(self,currnet,last_input_size):
        # in the last net
        # +1 for softmax
        if self.depth_size + 1 <= len(currnet) / 3:
            test_net = currnet.copy()
            test_net.extend([nn.Linear(last_input_size,self.label_size) ,nn.LogSoftmax()])
            run_test = Oven(self.input_size, self.optimizers_list,self.dropout, self.epoch,
                            self.learning_rate,self.batch_size)
            run_test.set_layers(test_net)
            #run_test.print_debug = True
            if self.dev != "cpu":
                run_test.use_gpu()
            accuracy, lr, optimazier = run_test.best_Values_for_model_without_droupout(self.train_loader,self.test_loader)
            self._check_new_result(accuracy, lr, optimazier,run_test)

        else:
            for next_size in self.features_size_to_test:
                test_net = currnet.copy()
                test_net.extend([nn.Linear(last_input_size,next_size) ,
                               nn.BatchNorm1d(next_size),nn.ReLU(inplace=True)])

                self._recursiv_build_network_to_test(test_net,next_size)

    def _check_new_result(self, accuracy,lr, optimazier,net):
        if accuracy > self.best_accuracy:
            print(f"found better params accuracy:{accuracy} lr:{lr} opt: {optimazier}")
            self.best_accuracy = accuracy
            pickle.dump(net, open(f"model_len{self.depth_size}_{accuracy}_accuracy.pkl", "wb"))
            # open(f"scheme_model_{accuracy}_accuracy.txt", "w").write(net.model_summery())

    def __printer(self,text, level = 0):
        if level < self.debug_level:
            print(text)



if __name__ =="__main__":
    test = Chef()
    test.find_best_arc()