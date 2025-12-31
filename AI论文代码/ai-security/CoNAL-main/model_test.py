
from utils import *
from torch.utils.data import DataLoader
from workflow import *
import random
from conal import *
from sklearn.metrics import confusion_matrix
seed = 12
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if __name__ == "__main__":

    dataset = 'labelme'
    model_dir = './model/'

    test_dataset = Dataset(mode='test', dataset=dataset)
    tst_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = torch.load(model_dir + 'model%s_conal_15' % dataset)
    # model.to("cuda" if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    target = []
    predict = []
    for _, inputs, targets in tst_loader:
        inputs = inputs.to("cuda" if torch.cuda.is_available() else 'cpu')
        target.extend(targets.data.numpy())
        targets = targets.to("cuda" if torch.cuda.is_available() else 'cpu')
        total += targets.size(0)
        output, _ = model(inputs, mode='test')
        _, predicted = output.max(1)
        predict.extend(predicted.cpu().data.numpy())
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    f1 = f1_score(target, predict, average='macro')
    classes = list(set(target))
    classes.sort()
    acc_per_class = []  # 存储每个类别的准确率
    predict = np.array(predict)
    target = np.array(target)
    err_record=[] #记录预测错误的情况【0】为真实标签，【1】为预测标签
    for i in range(len(target)):
        if predict[i]!=target[i]:
            # 记录预测错误的情况
            temp_arr=np.array([target[i],predict[i]])
            err_record.append(temp_arr)
    for i in range(len(classes)):
        instance_class = target == i
        acc_i = np.mean(predict[instance_class] == classes[i])
        acc_per_class.append(acc_i)
    # 存储预测的类型、分类准确率、F1分数和每个类别的准确率
    confusionMatrix = confusion_matrix(target, predict)
    np.save('./testResult/test_acc_conal_15.npy', acc)
    np.save('./testResult/test_f1_conal_15.npy', f1)
    np.save('./testResult/acc_per_class_conal_15.npy', acc_per_class)
    np.save('./testResult/test_predict_conal_15.npy', predict)
    np.save('./testResult/err_record_conal_15.npy', err_record)
    np.save('./testResult/confusionMatrix_conal_15.npy', confusionMatrix)
    np.save('./testResult/target_conal_15.npy', target)
    #
    # np.save('./testResult/test_acc_conal.npy', acc)
    # np.save('./testResult/test_f1_conal.npy', f1)
    # np.save('./testResult/acc_per_class_conal.npy', acc_per_class)
    # np.save('./testResult/test_predict_conal.npy', predict)
    # np.save('./testResult/err_record_conal.npy', err_record)
    # np.save('./testResult/confusionMatrix_conal.npy', confusionMatrix)
    # np.save('./testResult/target_conal.npy', target)

    # np.save('./testResult/test_acc_conal_8.npy', acc)
    # np.save('./testResult/test_f1_conal_8.npy', f1)
    # np.save('./testResult/acc_per_class_conal_8.npy', acc_per_class)
    # np.save('./testResult/test_predict_conal_8.npy', predict)
    # np.save('./testResult/err_record_conal_8.npy', err_record)
    # np.save('./testResult/confusionMatrix_conal_8.npy', confusionMatrix)
    # np.save('./testResult/target_conal_8.npy', target)

    print("test准确率为：{%.5f}" % acc)
