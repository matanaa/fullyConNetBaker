from gamesDataset import GamesDataset
from models import *

data_set_path = r"/home/dsi/matanak/bughouse/data/Features2006.csv"
data_set_path = r"/home/dsi/matanak/bughouse/data/Features2007.csv"
# pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/epoch_58_89.54000091552734/89.54000091552734.pkl" #2006 - 85.0976         Test set: Average loss: 0.3643, Accuracy: 382165/449090 (85%)
pickle_file = r"/home/dsi/matanak/bughouse/90.0.pkl"  # 2006 -  85.5532       Test set: Average loss: 0.3452, Accuracy: 384211/449090 (86%), 2007 85.3605         Test set: Average loss: 0.3484, Accuracy: 372433/436306 (85%)
# pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/89.04000091552734_epoch_19/89.04000091552734.pkl" #2006 -  85.5532       Test set: Average loss: 0.3452, Accuracy: 384211/449090 (86%) , 2007 - 85.7002 Test set: Average loss: 0.3373, Accuracy: 373915/436306 (86%)
# pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/89.79999542236328_epoch_41/89.79999542236328.pkl" #2006 - 85.7118        Test set: Average loss: 0.3423, Accuracy: 384918/449084 (86%) , 2007 - 85.5186 Test set: Average loss: 0.3460, Accuracy: 373123/436306 (86%)
# pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/86.19999694824219_epoch_36/86.19999694824219.pkl" #2006 - 86.4079         Test set: Average loss: 0.3143, Accuracy: 388044/449084 (86%), 2007 - 86.2663 Test set: Average loss: 0.3180, Accuracy: 376385/436306 (86%)


# pickle_file = r"C:\Users\matan.LIACOM\Desktop\test\model_len3_85.80199432373047_accuracy.pkl"
# data_set_path = r"C:\Users\matan.LIACOM\PycharmProjects\bughouse-partner-matching\data\fake_20_20.csv"
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

print(net.validate(test_loader))
print(pickle_file)
# print(net.model_summery())
