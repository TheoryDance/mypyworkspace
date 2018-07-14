import os, sys
sys.path.append(r'C:\Users\Administrator\git\mypyworkspace')

def unpickle(file):  # use cifar extract data
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_meta = unpickle('batches.meta')

label_names = batch_meta[b'label_names']
for name in label_names:
    print(name)