import torch
import matplotlib.pyplot as plt

def normtest(f, param):
    x = torch.linspace(-10000,10000,10000000)
    for i in range(len(param)):
        param[i] = torch.ones_like(x)*param[i]
    y = f(x, *param)
    area = torch.sum(y)*(torch.max(x)-torch.min(x))/len(x)
    plt.plot(x, y)
    plt.yscale('log')
    plt.show()
    return area

def doublenormtest(f, param1, param2):
    x1 = torch.linspace(-10000,10000,10000000)
    x2 = torch.linspace(-10000,10000,10000000)
    x = torch.cat((x1, x2))
    
    ps = []
    for i in range(len(param1)):
        p1 = torch.ones(len(x1))*param1[i]
        p2 = torch.ones(len(x2))*param2[i]
        p = torch.cat((p1,p2))
        ps.append(p)

    y = f(x, *ps)
    area = torch.sum(y)*(torch.max(x)-torch.min(x))/len(x)

    plt.plot(x, y)
    plt.yscale('log')
    plt.show()
    return area

