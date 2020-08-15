from gamesDataset import GamesDataset
from models import *
data_set_path = r"../bughouse-partner-matching/data/Features2005p1.csv"
test = GamesDataset(data_set_path)
test_set_len = 50000
train_set, val_set = torch.utils.data.random_split(test, [len(test)-test_set_len,test_set_len])
batch_size = 400
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)



finder = Chef(train_loader, test_loader, optimizer={"AdaDelta":optim.Adadelta},batch_size=batch_size,
              epoch=1)

finder.set_features_size_list([256,512,1024,2046,52,52*2,52*4,52*6,51,51*2])
finder.set_network_length(3)
finder.find_best_arc()
