import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # CUDA 4
import argparse
import torch.nn as nn
from model import Generator
from data_process import prepro
from data_process import prepro_valid
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np


def train_network(train_loader, optimizer, net, criterion, opt):
    net = net.train()
    train_loss = 0.0
    train_len = 0.0
    train_correct = 0.0

    for epoch_batch, (src, tgt) in enumerate(train_loader):
        src = src.type(torch.FloatTensor)
        src = src.to(opt.device)
        tgt = tgt.squeeze(1)
        tgt = tgt.type(torch.LongTensor)
        tgt = tgt.to(opt.device)

        optimizer.zero_grad()
        output = net(src)
        loss = criterion(output, tgt)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        train_len += tgt.shape[0]
        pred = torch.max(output, 1)[1]
        train_correct += pred.eq(tgt.data.view_as(pred)).cpu().sum()

    Train_loss = train_loss / train_len
    Train_loss_all = train_loss
    Train_correct = train_correct / train_len
    return Train_loss, Train_loss_all, Train_correct


def valid_network(valid_loader, net, criterion, opt):
    net = net.eval()  # 改动 net.train()
    valid_loss = 0.0
    valid_len = 0.0
    valid_correct = 0.0

    with torch.no_grad():
        for epoch_batch, (src, tgt) in enumerate(valid_loader):
            src = src.type(torch.FloatTensor)
            src = src.to(opt.device)
            tgt = tgt.squeeze(1)
            tgt = tgt.type(torch.LongTensor)
            tgt = tgt.to(opt.device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = net(src)
            loss = criterion(outputs, tgt)
            valid_loss += loss.item()
            valid_len += outputs.shape[0]
            pred = torch.max(outputs, 1)[1]
            valid_correct += pred.eq(tgt.data.view_as(pred)).cpu().sum()

        Valid_loss = valid_loss / valid_len
        Valid_loss_all = valid_loss
        Valid_correct = valid_correct / valid_len
        print('验证损失函数为{}, 总损失函数为{}, 验证准确率{}'.format(Valid_loss, Valid_loss_all, Valid_correct))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default=r'F:\LYY\dataset\Elec_iden\个体识别数据集\train\train')
    parser.add_argument('--valid_data', default=r'F:\LYY\dataset\Elec_iden\个体识别数据集\validation\validation')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init', default='normal', type=str, help='parameter initializer to use')
    parser.add_argument('--init_std', type=float, default=0.02, help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.001, help='parameters initialized by N(0, init_std)')
    parser.add_argument('--dropout', default=0.0)

    parser.add_argument('--batch_size', default=2048)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--seq_len_in', default=3000)
    parser.add_argument('--resolution', default=500)
    parser.add_argument('--n_heads', default=4)
    parser.add_argument('--mlp_ratio', default=4)
    parser.add_argument('--num_classes', default=10)

    opt = parser.parse_args()

    def init_weight(weight):
        if opt.init == 'uniform':
            nn.init.uniform_(weight, -opt.init_range, opt.init_range)
        elif opt.init == 'normal':
            nn.init.normal_(weight, 0.0, opt.init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('Softmax') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, opt.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)

    train_src, train_trg = prepro(d_path_train=opt.train_data)
    valid_src, valid_trg = prepro_valid(d_path_validation=opt.valid_data)
    print('数据装载...')

    train_data = TensorDataset(train_src, train_trg)
    valid_data = TensorDataset(valid_src, valid_trg)

    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=opt.batch_size)
    valid_loader = DataLoader(dataset=valid_data, shuffle=True, batch_size=opt.batch_size)
    opt.device = torch.device('cpu')

    net = Generator(n_heads=opt.n_heads, mlp_ratio=opt.mlp_ratio, resolution=opt.resolution,
                    num_classes=opt.num_classes, dropout=0.)

    net = net.to(opt.device)
    net.apply(weights_init)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print('开始训练...')
    print('resolution = {}'.format(opt.resolution))
    for epoch in range(opt.epochs):
        print('第{}轮'.format(epoch))
        train_loss, train_loss_all, Train_correct = train_network(train_loader, optimizer, net, criterion, opt)
        print('平均损失函数{}, 损失函数{}, 训练准确率{}'.format(train_loss, train_loss_all, Train_correct))
        valid_network(valid_loader, net, criterion, opt)


if __name__ == '__main__':
    torch.manual_seed(13)
    np.random.seed(13)
    print(torch.cuda.is_available())
    main()
