import os
from fnmatch import fnmatch

from PairMeansDataset import PairMeansDataset
from models import *

def findPKL(root='/home/dsi/matanak/fullyConNetBaker/results/89.98307037353516_epoch_104', pattern="*.pkl"):
    ret_list =[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                ret_list.append(os.path.join(path, name))
    return ret_list

data_set_path = r"/home/dsi/matanak/bughouse/data/pairs_dataset_2006.csv"
pickle_file = r"/home/dsi/matanak/fullyConNetBaker/results/89.98307037353516_epoch_104/89.98307037353516.pkl"  # 2006 - 86.4079         Test set: Average loss: 0.3143, Accuracy: 388044/449084 (86%), 2007 - 86.2663 Test set: Average loss: 0.3180, Accuracy: 376385/436306 (86%)





# pickle_file = r"C:\Users\matan.LIACOM\Desktop\test\model_len3_85.80199432373047_accuracy.pkl"
# data_set_path = r"C:\Users\matan.LIACOM\PycharmProjects\bughouse-partner-matching\data\fake_20_20.csv"

test_dataset = PairMeansDataset(data_set_path)
if torch.cuda.is_available() :
    print(f"[!]Loading Data to GPU, data file {data_set_path}")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=99999, shuffle=False, num_workers=60, pin_memory=True)

else:
    print(f"[!]Loading Data to RAM, data file {data_set_path}")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


max_acr = 0
max_pickle_file =""
for pickle_file in findPKL(root=r"/home/dsi/matanak/fullyConNetBaker/results/"):
    net = pickle.load(open(pickle_file, "rb"))
    if torch.cuda.is_available():
        dev = "cuda"
        dtype = torch.cuda.FloatTensor
        net.use_gpu(True)
        # net.Net.cuda()

        print("[!]using cuda GPU")
    else:
        net.use_gpu(False)
        dev = "cpu"
        print("[!]using CPU")
    acr = net.validate(test_loader)
    print(pickle_file,acr)
    if acr> max_acr:
        max_pickle_file = pickle_file
        max_acr=acr
print(f"\n\npickel: {max_pickle_file} accurecy: {max_acr} ")


# print(net.model_summery())
