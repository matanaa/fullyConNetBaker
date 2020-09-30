from gamesDataset import GamesDataset
from models import *

data_set_path = r"/home/dsi/matanak/bughouse/data/Features2006.csv"
test_dataset = GamesDataset(data_set_path)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/epoch_58_89.54000091552734/89.54000091552734.pkl"
net = pickle.load(open(pickle_file, "rb"))

net.use_gpu(False)
print (net.validate(test_loader))
print(net.model_summery())