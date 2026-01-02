
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
        while numpy.sum(1 - data[4, offset:offset + 1024]) < 256:
            offset = numpy.random.randint(1024)
        data = torch.FloatTensor(data[:, offset:offset + 1024])
        base = gaussian_blur(data[None, None, None, 1], (13, 1))[0, 0]
        target = data[0] - base
        target = torch.sign(target) * (target.abs() + 1).log() / 2
        return data[2:-1], target, base, data[-1:]

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
    for i in range(6):
        test, target, base, mask = dataset[-i]
        # for k in range(test.shape[0]):
        #     pyplot.plot(test[k])
        # pyplot.plot()
        pyplot.subplot(2, 3, i + 1)
        pyplot.plot(target[0] + base[0])
        pyplot.plot(base[0])
        pyplot.plot(mask[0])
    print(test.shape, target.shape, base.shape, mask.shape)
    pyplot.show()
