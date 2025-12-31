import torch


def multi_loss(y_true, y_pred, loss_fn=torch.nn.CrossEntropyLoss(reduction='mean').to("cuda"if torch.cuda.is_available() else 'cpu')):
    mask = y_true != -1
    y_pred = torch.transpose(y_pred, 1, 2)
    loss = loss_fn(y_pred[mask], y_true[mask])
    return loss