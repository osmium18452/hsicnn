import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from HSIPreprocess import HSIPreprocess, HSIDataset
from CNN import CNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-p', '--patch_size', type=int, default=7)
    parser.add_argument('-r', '--ratio', type=float, default=.3)
    parser.add_argument('-c', '--CUDA', action='store_true')
    parser.add_argument('-l', '--lr', type=float, default=0.0001)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    args = parser.parse_args()

    batch_size = args.batch_size
    patch_size = args.patch_size
    ratio = args.ratio
    CUDA = args.CUDA
    lr = args.lr
    epoch = args.epoch

    dataset = HSIPreprocess('pc', patch_size, ratio)
    train_set=HSIDataset(dataset.train_set,dataset.train_labels)
    test_set=HSIDataset(dataset.test_set,dataset.test_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    cnn=CNN(bands=dataset.bands,patch_size=patch_size,num_classes=dataset.num_classes)

    if CUDA:
        cnn.cuda()
    optimizer=torch.optim.Adam(cnn.parameters(),lr=lr)
    loss_func=torch.nn.CrossEntropyLoss()
    pbar_epoch = tqdm(total=epoch, ascii=True, dynamic_ncols=True)
    for i in range(epoch):
        pbar_iter = tqdm(total=len(train_loader), ascii=True, dynamic_ncols=True, leave=False)
        for step,(x,y) in enumerate(train_loader):
            if CUDA:
                x=x.cuda()
                y=y.cuda()
            output=cnn(x)
            loss=loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_iter.set_postfix_str('loss: %.4f' % loss.item())
            pbar_iter.update()
        pbar_iter.close()
        pbar_epoch.set_postfix_str('train loss: %.4f' % (loss.item()))
        pbar_epoch.update()

