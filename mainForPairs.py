import torch
import torch.optim as optim

from PairMeansDataset import PairMeansDataset
from models import *

# data_set_path = r"../bughouse-partner-matching/data/Features2005p1.csv"
data_set_path = r"../bughouse/data/pairs_dataset_2005.csv"
# data_set_path = r"C:\Users\matan.LIACOM\PycharmProjects\bughouse-partner-matching\data\pairs_dataset_2005.csv"
test = PairMeansDataset(data_set_path)
test_set_len = int(len(test)/10)
train_set, val_set = torch.utils.data.random_split(test, [len(test) - test_set_len, test_set_len])
batch_size = 500

if torch.cuda.is_available() :

    print(f"[!]Loading Data to GPU, data file {data_set_path}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,num_workers=60, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=60, pin_memory=True)
else:
    print(f"[!]Loading Data to RAM, data file {data_set_path}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

finder = Chef(train_loader, test_loader, optimizer={"AdaDelta": optim.Adadelta}, batch_size=batch_size, epoch=1,
              input_size=84)

finder.set_features_size_list([256, 512, 1024, 2048, 52, 52 * 2, 52 * 4, 52 * 6, 51, 51 * 2])
finder.set_features_size_list([256, 512, 1024, 52 * 8, 52 * 2, 52 * 6])
finder.set_features_size_list([512, 256, 128])
finder.set_network_length(4)
finder.use_gpu()
finder.find_best_arc()
