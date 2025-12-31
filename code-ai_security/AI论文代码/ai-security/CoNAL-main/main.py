import numpy as np

from conal import *
import copy
from utils import *
from torch import optim
from copy import deepcopy
import argparse
from options import *
from torch.utils.data import DataLoader
from workflow import *
import random
from conal import *
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score

seed = 12
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = 'labelme'
model_dir = './model/'

train_dataset = Dataset(mode='train', dataset=dataset, sparsity=0)
trn_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
valid_dataset = Dataset(mode='valid', dataset=dataset)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_dataset = Dataset(mode='test', dataset=dataset)
tst_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def main(opt, model=None):
    train_acc_list = []
    test_acc_list = []
    test_f1_list = []
    valid_acc_list = []
    valid_f1_list = []
    user_feature = np.eye(train_dataset.num_users)
    if model == None:
        model = torch.load(model_dir + 'model%s' % dataset)
    else:
        model = CoNAL(num_annotators=train_dataset.num_users, num_class=train_dataset.num_classes,
                      input_dims=train_dataset.input_dims, user_feature=user_feature, gumbel_common=False,backbone_model=None).to("cuda"if torch.cuda.is_available() else 'cpu')
    best_valid_acc = 0
    best_model = None
    best_model_wts = None
    lr = 0.01 #0.01
    for epoch in range(350):  # opt.num_epochs
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_acc = train(train_loader=trn_loader, model=model, optimizer=optimizer, criterion=multi_loss,
                          # mode='other')
                          # mode='xiaorong')
                        mode='common')
                        # mode='other')

        valid_acc, valid_f1, _ = test(model=model, test_loader=val_loader)
        test_acc, test_f1, _ = test(model=model, test_loader=tst_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        valid_acc_list.append(valid_acc)
        test_f1_list.append(test_f1)
        valid_f1_list.append(valid_f1)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)
            best_model_wts = copy.deepcopy(model.state_dict())
        print('Epoch {0}------------------\n'.format(epoch))
        print('Train acc: %.5f\tValid acc: %.5f\tTest acc:%.5f' % (train_acc,valid_acc,test_acc))
    print('Best Valid acc: %.5f\n' % best_valid_acc)
    # print('Test acc: %.5f, Test f1: %.5f\n' % (test_acc, test_f1))
    # 写入数据
    #
    # np.save("./trainProcess/train_acc_conal_15.npy",train_acc_list)
    # np.save("./trainProcess/valid_acc_conal_15.npy", valid_acc_list)
    # np.save("./trainProcess/valid_f1_conal_15.npy", valid_f1_list)
    # np.save("./trainProcess/test_acc_conal_15.npy", test_acc_list)
    # np.save("./trainProcess/test_f1_conal_15.npy", test_f1_list)

    # np.save("./trainProcess/train_acc_conal_8.npy",train_acc_list)
    # np.save("./trainProcess/valid_acc_conal_8.npy", valid_acc_list)
    # np.save("./trainProcess/valid_f1_conal_8.npy", valid_f1_list)
    # np.save("./trainProcess/test_acc_conal_8.npy", test_acc_list)
    # np.save("./trainProcess/test_f1_conal_8.npy", test_f1_list)

    # np.save("./trainProcess/train_acc_conal.npy",train_acc_list)
    # np.save("./trainProcess/valid_acc_conal.npy", valid_acc_list)
    # np.save("./trainProcess/valid_f1_conal.npy", valid_f1_list)
    # np.save("./trainProcess/test_acc_conal.npy", test_acc_list)
    # np.save("./trainProcess/test_f1_conal.npy", test_f1_list)

    # torch.save(best_model_wts, "./model/modellabelme.pth")  # 保存最佳模型的参数
    # torch.save(best_model, "./model/modellabelme_conal_15")  # 保存最佳模型的参数
    test_acc_best, test_f1, _ = test(model=best_model, test_loader=tst_loader)
    print('--------------------------------------------------')
    print('Best Test acc: %.5f, Test f1: %.5f' % (test_acc_best, test_f1))
    return best_model, test_acc_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_opts(parser)
    opt = parser.parse_args()

    test_acc = []
    _, acc = main(opt, model=True)
