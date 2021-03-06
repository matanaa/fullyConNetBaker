# from MeansDataset import MeansDataset
import pickle

from gamesDataset import GamesDataset
# from models_old import *
from models import *

pickle_file = r"model_len3_85.80199432373047_accuracy.pkl"
pickle_file = r"model_len20_85.78599548339844_accuracy.pkl"
data_set_path = r"../bughouse/data/Features2005.csv"

# pickle_file = r"C:\Users\matan.LIACOM\Desktop\test\model_len3_85.80199432373047_accuracy.pkl"
# data_set_path = r"C:\Users\matan.LIACOM\PycharmProjects\bughouse-partner-matching\data\fake_20_20.csv"
print("[!]Loading Model")
net = pickle.load(open(pickle_file, "rb"))

test = GamesDataset(data_set_path)
test_set_len = 5000
train_set, val_set = torch.utils.data.random_split(test, [len(test) - test_set_len, test_set_len])
batch_size = 600
print("[!]Loading Data")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available() and False:
    dev = "cuda"
    dtype = torch.cuda.FloatTensor
    net.use_gpu(True)
    # net.Net.cuda()

    print("[!]using cuda GPU")
else:
    net.use_gpu(False)
    dev = "cpu"
    print("[!]using CPU")

net.optimizer = optim.Adadelta(net.Net.parameters(), lr=net.learning_rate)
# net.print_debug = True
net.set_epoch(45)

# net.total_loss=[]
# net.total_accuracy=[]
net.train_and_vaildate(train_loader, test_loader)
