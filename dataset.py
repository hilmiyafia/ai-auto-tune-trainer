
import os
import numpy
import torch
from torchvision.transforms.functional import gaussian_blur

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path="npys"):
        self.files = [f"{path}/{file}" for file in os.listdir(path)]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = numpy.load(self.files[index])
        offset = numpy.random.randint(1024)
        while numpy.sum(1 - data[4, offset:offset + 1024]) < 0.5:
            offset = numpy.random.randint(1024)
        data = torch.FloatTensor(data[:, offset:offset + 1024])
        base = gaussian_blur(data[None, None, None, 1], (13, 1))[0, 0]
        return data[2:], data[0] - base, base

def get_dataloader(dataset, batch_size=8, shuffle=True, drop_last=True):
    return torch.utils.data.DataLoader(
        dataset,
        drop_last=drop_last,
        batch_size=batch_size, 
        shuffle=shuffle,  
        num_workers=4, 
        persistent_workers=True)

if __name__ == "__main__":
    import matplotlib.pyplot as pyplot
    dataset = Dataset()
    test, target, base = dataset[0]
    for k in range(test.shape[0]):
        pyplot.plot(test[k])
    pyplot.plot(base[0])
    pyplot.plot(target[0])
    pyplot.show()
