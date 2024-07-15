from torch.nn.functional import sigmoid
from torch.nn import BCELoss

crossloss = BCELoss()

def classifier_loss(pred, target, weight=None):
    batch_size = pred.size()[0]

    prob = sigmoid(pred)

    return crossloss(prob, target)
    
