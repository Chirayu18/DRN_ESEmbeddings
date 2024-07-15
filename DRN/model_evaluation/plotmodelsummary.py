import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

matplotlib.use('pgf')
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({
    'font.size': 30,         # Main text font size
    'axes.labelsize': 20,    # X and Y axis label font size
    'xtick.labelsize': 20,   # X-axis tick labels font size
    'ytick.labelsize': 20,   # Y-axis tick labels font size
    'legend.fontsize': 20,  # Legend font size
    'figure.titlesize': 50,  # Figure title font size
})

pred = np.array(np.load("pred.pickle",allow_pickle=True))
true = np.array(np.load("trueE_target.pickle",allow_pickle=True))
file=open("all_trainidx.pickle",'rb')
tidx=np.array(pickle.load(file))
file=open("all_valididx.pickle",'rb')
vidx=np.array(pickle.load(file))

predv=pred[vidx]
predt=pred[tidx]
truev=true[vidx]
truet=true[tidx]

modelparams,title = "",""
with open('params.txt','r') as file:
    title = file.readline()
    modelparams = file.read()


fig,axs=plt.subplots(2,2,figsize=(32,18),gridspec_kw={'height_ratios': [1.8, 1]})
h = axs[0,0].hist2d(np.array(predt),np.array(truet),bins=50,range=[[0.1,2],[0.1,2]]) 
plt.colorbar(h[3],ax=axs[0,0],label="NEvents")
axs[0,0].set_title("Training sample ( " + str(len(predt)) + " events )")
axs[0,0].set_xlabel("Predicted")
axs[0,0].set_ylabel("True")
h = axs[0,1].hist2d(np.array(predv),np.array(truev),bins=50,range=[[0.1,2],[0.1,2]]) 
plt.colorbar(h[3],ax=axs[0,1],label="NEvents")
axs[0,1].set_title("Validation sample ( " + str(len(predv)) + " events )")
axs[0,1].set_xlabel("Predicted")
axs[0,1].set_ylabel("True")
fig.suptitle(title,fontsize=30)
plt.savefig("predvstrue.png")

s = np.load("summaries.npz")
tloss=s['train_loss']
vloss=s['valid_loss']
epoch=s['epoch']
axs[1,0].plot(epoch,tloss,'-',label="Training Loss")
axs[1,0].plot(epoch,vloss,'-',label="Validation Loss")
x,y = np.argmin(vloss),np.min(vloss)
axs[1,0].axhline(y, color='r', linestyle='--',label='Best model\nMin Loss: '+str(y)+', Epoch: '+str(x))
axs[1,0].axvline(x, color='r', linestyle='--')
#plt.xticks(list(plt.xticks()[0]) + [x])
#plt.yticks(list(plt.yticks()[0]) + [y])
#plt.title(title)
axs[1,0].set_xlabel("Epoch")
axs[1,0].set_ylabel("Loss")
#axs[1,0].savefig("loss.png")
axs[1,0].legend()
plt.rcParams.update({"text.usetex": True})
axs[1,1].text(0,1,modelparams,ha="left",va="top",bbox=dict(boxstyle = "square",
                  facecolor = "white"),fontsize=35)
axs[1,1].set_axis_off()
fig.savefig("model_summary.pdf",bbox_inches='tight')
fig.savefig("model_summary.png",bbox_inches='tight')
