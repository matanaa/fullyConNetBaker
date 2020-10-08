from gamesDataset import GamesDataset
from models import *

data_set_path = r"/home/dsi/matanak/bughouse/data/Features2006.csv"
# pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/epoch_58_89.54000091552734/89.54000091552734.pkl" #85.0976         Test set: Average loss: 0.3643, Accuracy: 382165/449090 (85%)
pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/epoch_19_90.0/90.0.pkl"                              # 85.5532       Test set: Average loss: 0.3452, Accuracy: 384211/449090 (86%)

pickle_file = r"C:\Users\matan.LIACOM\Desktop\test\model_len3_85.80199432373047_accuracy.pkl"
data_set_path = r"C:\Users\matan.LIACOM\PycharmProjects\bughouse-partner-matching\data\fake_20_20.csv"
net = pickle.load(open(pickle_file, "rb"))
test_dataset = GamesDataset(data_set_path)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
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


print (net.validate(test_loader))
# print(net.model_summery())