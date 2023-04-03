import torch
import argparse
import torch.nn as nn
from model import Generator
from data_process import prepro_test
from torch.utils.data import DataLoader
import numpy as np


def test_model(net, test_loader, criterion, opt):
    net.eval()
    with torch.no_grad():
        for epoch_batch, (src) in enumerate(test_loader):
            src = src.type(torch.FloatTensor)
            src = src.to(opt.device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = net(src)
            predict = torch.max(outputs, dim=1)[1]
            predict = predict.tolist()
            file_output = open("test/output.txt",'w')
            for i in predict:
                file_output.write(str(i+1)+'\n')
            file_output.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default=r'test')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init', default='normal', type=str, help='parameter initializer to use')
    parser.add_argument('--init_std', type=float, default=0.02, help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.001, help='parameters initialized by N(0, init_std)')
    parser.add_argument('--dropout', default=0.0)

    parser.add_argument('--batch_size', default=2048)
    parser.add_argument('--epochs', default=1)
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
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
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

    test_src = prepro_test(d_path_test=opt.test_data)
    test_loader = DataLoader(dataset=test_src, shuffle=False, batch_size=opt.batch_size)
    opt.device = torch.device('cpu')


    net = Generator(n_heads=opt.n_heads, mlp_ratio=opt.mlp_ratio, resolution=opt.resolution,
                    num_classes=opt.num_classes, dropout=0.)

    net = net.to(opt.device)
    net.apply(weights_init)

    net.load_state_dict(torch.load(r'result.model', map_location='cpu'))
    criterion = nn.CrossEntropyLoss()

    print('start test model ...')
    test_model(net, test_loader, criterion, opt)


if __name__ == '__main__':
    torch.manual_seed(13)
    np.random.seed(13)
    main()
