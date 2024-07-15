import torch
cf = torch.load('cartfeat_ES.pickle')
for i in cf:
	i.x = torch.cat((i.xECAL,i.xES),0)
torch.save(cf,'cartfeat.pickle')
