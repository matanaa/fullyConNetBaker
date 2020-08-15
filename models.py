import torch
dtype = torch.float
device = torch.device("cpu")
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, input_size=28 * 28, optimizer=None, dropout=None, epoch=10, learning_rate=0.001, batch_size=100):
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

        self.lr_start = 0.05
        self.lr_end= 0.5
        self.lr_step = 0.05
        if not optimizer:
            self.optimizers_list = {"SGD": optim.SGD, "ADAM": optim.Adam, "RMSprop": optim.RMSprop, "AdaDelta": optim.Adadelta}
        else:
            self.optimizers_list = optimizer

        self.optimizer = None




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
            output = self.Net.forward(Variable(data))
            loss = F.nll_loss(output, labels)
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
                output = self.Net(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).cpu().sum()
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
        for epoch_i in range(self.epoch):
            print(f"[!]Train Epoch: {epoch_i}")
            self.step_train(train_loader, epoch_i)
            self.validate(test_loader)
            torch.save(self.Net.state_dict(), f"save_State{epoch_i}.tourch")

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

    def showGraphs(self):
        """
        show the required graphs
        :return:
        """
        import matplotlib.pyplot as plt
        plt.plot(range(1,self.epoch+1), self.total_loss, label=f'Loss - {self.Net.name()} ')
        plt.legend(bbox_to_anchor=(1.0, 1.00))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.xticks(plt_learning)
        plt.show()

        plt.plot(range(1,self.epoch+1), self.total_accuracy, label=f'Accuracy - {self.Net.name()} ')
        plt.legend(bbox_to_anchor=(1.0, 1.00))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.xticks(plt_learning)
        plt.show()

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

                net_to_check = model(layers=1)
                net_to_check.step_train(train_loader)
                print(f"[+]test on:lr: {lr} optimaize: {optimazier_name}")
                curr_accuracy = net_to_check.validate(test_loader)
                if best_accuracy < curr_accuracy:
                    best_accuracy = curr_accuracy
                    best_lr = lr
                    best_optimazier = optimazier_name
                    print(
                        f"[!!!]found better parrams:  accuracy: {best_accuracy} lr: {lr} optimaize: {best_optimazier} ")
        print(best_accuracy, best_lr, best_optimazier)
        return (best_accuracy, best_lr, best_optimazier)


class Chef:
    """
    class that manage the network
    """
    def __init__(self,train_loader, test_loader, input_size=28 * 28, optimizer=None, dropout=None, epoch=10, learning_rate=0.001, batch_size=100):
        """
        init the object
        :param input_size:
        :param optimizer: which oprimaizer to run
        :param dropout: if it use dropout, the dropout rate
        :param epoch:
        :param learning_rate:
        :param batch_size:
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_size = input_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_debug = False

        if not optimizer:
            self.optimizers_list = {"SGD": optim.SGD, "ADAM": optim.Adam, "RMSprop": optim.RMSprop, "AdaDelta": optim.Adadelta}
        else:
            self.optimizers_list = optimizer

        self.dropout = dropout

        self.debug_level = 10

        self.depth_size = 3
        self.features_size_to_test = [52,256,512,1024,2046,52*2,52*4,52*6]
        self.features_size_to_test = [52,256]
        self.label_size = 2

    def find_best_arc(self):
        for next_size in self.features_size_to_test:
            first = [nn.Linear(self.input_size,next_size) ,
                               nn.BatchNorm2d(next_size),nn.ReLU(inplace=True)]
            self._recursiv_build_network_to_test(first,next_size)


    def _recursiv_build_network_to_test(self,currnet,last_input_size):
        # in the last net
        # +1 for softmax
        if self.depth_size + 1 <= len(currnet) / 3:
            test_net = currnet.copy()
            test_net.extend([nn.Linear(last_input_size,self.label_size) ,nn.LogSoftmax()])
            run_test = Oven(self.input_size, self.optimizers_list,self.dropout, self.epoch,
                            self.learning_rate,self.batch_size)

            run_test.best_Values_for_model_without_droupout(self.train_loader,self.test_loader)


        else:
            for next_size in self.features_size_to_test:
                test_net = currnet.copy()
                test_net.extend([nn.Linear(last_input_size,next_size) ,
                               nn.BatchNorm2d(next_size),nn.ReLU(inplace=True)])

                self._recursiv_build_network_to_test(test_net,next_size)


    def __printer(self,text, level = 0):
        if level < self.debug_level:
            print(text)



if __name__ =="__main__":
    test = Chef()
    test.find_best_arc()