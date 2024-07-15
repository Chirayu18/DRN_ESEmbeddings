import uproot
import math
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from time import time
import pickle
import tqdm
from torch_geometric.data import Data
import torch

MISSING = -999999

########################################
# HGCAL Values                         #
########################################

HGCAL_X_Min = -36
HGCAL_X_Max = 36

HGCAL_Y_Min = -36
HGCAL_Y_Max = 36

HGCAL_Z_Min = 13
HGCAL_Z_Max = 265

HGCAL_Min = 0
HGCAL_Max = 230
# HGCAL_Max_EE =3200
# HGCAL_Max_FH = 2200
# HGCAL_MAX_AH =800


########################################
# ECAL Values                          #
########################################

X_Min = -150
X_Max = 150

Y_Min = -150
Y_Max = 150

Z_Min = -330
Z_Max = 330

Eta_Min = -1.4
Eta_Max = 1.4

Phi_Min = -np.pi
Phi_Max = np.pi

iEta_Min = -85
iEta_Max = 85

iPhi_Min = 1
iPhi_Max = 360

iX_Min = 1
iX_Max = 100

iY_Min = 1
iY_Max = 100

ECAL_Min = 0
ECAL_Max = 100

def detfeat(ieta,iphi,energy):
	e = rescale(ieta,iEta_Min,iEta_Max)
	p = rescale(iphi,iPhi_Min,iPhi_Max)
	en = rescale(energy,ECAL_Min,ECAL_Max)
	return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
            ),
            -1,
        )

def localfeat(eta,phi,energy,energy2=None):
	
	e = rescale(eta,Eta_Min,Eta_Max)
	p = rescale(phi,Phi_Min,Phi_Max)
	en = rescale(energy,ECAL_Min,ECAL_Max)
	#en2 = rescale(energy2,ECAL_Min,ECAL_Max)
	return ak.concatenate(
            (
                e[:, :, None],
                p[:, :, None],
                en[:, :, None],
	#	en2[:,:,None],
            ),
            -1,
        )
	
def setptetaphie(pt, eta, phi, e):
    pta = abs(pt)
    return np.array([np.cos(phi) * pta, np.sin(phi) * pt, np.sinh(eta) * pta, e])


def getMass(lvec):
    return np.sqrt(
        lvec[3] * lvec[3] - lvec[2] * lvec[2] - lvec[1] * lvec[1] - lvec[0] * lvec[0]
    )


def rescale(feature, minval, maxval):
    top = feature - minval
    bot = maxval - minval
    return top / bot


def dphi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    gt = dphi > np.pi
    dphi[gt] = 2 * np.pi - dphi[gt]
    return dphi


def dR(eta1, eta2, phi1, phi2):
    dp = dphi(phi1, phi2)
    de = np.abs(eta1 - eta2)

    return np.sqrt(dp * dp + de * de)


def cartfeat(x, y, z, En, det=None):
    E = rescale(En, ECAL_Min, ECAL_Max)
    x = rescale(x, X_Min, X_Max)
    y = rescale(y, Y_Min, Y_Max)
    z = rescale(z, Z_Min, Z_Max)


    if det is None:
        return ak.concatenate(
            (x[:, :, None], y[:, :, None], z[:, :, None], E[:, :, None]),
            -1,
        )
    else:
        return ak.concatenate(
            (
                x[:, :, None],
                y[:, :, None],
                z[:, :, None],
                E[:, :, None],
                E2[:, :, None],
            ),
            -1,
        )


def torchify(feat, graph_x=None):
    data = [
        Data(x=torch.from_numpy(ak.to_numpy(Pho).astype(np.float32))) for Pho in feat
    ]
    if graph_x is not None:
        for d, gx in zip(data, graph_x):
            d.graph_x = gx
    return data


def npify(feat):
    t0 = time()
    data = [ak.to_numpy(Pho) for Pho in feat]
    print("took %f" % (time() - t0))
    return data


class Extract:
    def __init__(
        self,
        outfolder="pickles",
        path="merged.root",
        treeName="nTuplelize/T",
        path2=None,
    ):
        if path is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree = uproot.open("%s:%s" % (path, treeName))
        if path2 is not None:
            # path = '~/shared/nTuples/%s'%path
            self.tree2 = uproot.open("%s:%s" % (path2, treeName))

        self.outfolder = outfolder

    def read(self, N=None):
        varnames = [
            "Hit_X_Pho1",
            "Hit_Y_Pho1",
            "Hit_Z_Pho1",
		"Hit_Eta_Pho1",
		"Hit_Phi_Pho1",
		"Hit_iEta_Pho1",
		"Hit_iPhi_Pho1",
            "RecHitEnPho1",
            "RecHitEZPho1",
            "RecHitETPho1",
            "Hit_X_Pho2",
            "Hit_Y_Pho2",
            "Hit_Z_Pho2",
		"Hit_Eta_Pho2",
		"Hit_Phi_Pho2",
		"Hit_iEta_Pho2",
		"Hit_iPhi_Pho2",
            "RecHitEnPho2",
            "RecHitEZPho2",
            "RecHitETPho2",
            "m_gen",
            "p_gen",
            "pt_gen",
            "eta_gen",
            "phi_gen",
		#"EEFlag1",
		#"EEFlag2",
        ]

        arrs = self.tree.arrays(varnames)
        arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho1"]]]
        arrs = arrs[[len(j) > 0 for j in arrs["Hit_X_Pho2"]]]
        result = {}
        t0 = time()
        print("\tDumping target took %f seconds" % (time() - t0))
        print("Building cartesian features..")

        target = arrs["m_gen"]
        #target = np.log(np.abs(target))
        with open("%s/trueE_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(target, outpickle)
        #weights = np.array([ 19*(int(len(j)>0)) + 1 for j in arrs["Hit_X_Pho2"]])
        #with open("tworeco_weights.pickle",'wb') as outpickle:
        #    pickle.dump(weights,outpickle)
        HitsX = ak.concatenate((arrs["Hit_X_Pho1"], arrs["Hit_X_Pho2"]), axis=1)
        HitsY = ak.concatenate((arrs["Hit_Y_Pho1"], arrs["Hit_Y_Pho2"]), axis=1)
        HitsZ = ak.concatenate((arrs["Hit_Z_Pho1"], arrs["Hit_Z_Pho2"]), axis=1)
        HitsEta = ak.concatenate((arrs["Hit_Eta_Pho1"], arrs["Hit_Eta_Pho2"]), axis=1)
        HitsPhi = ak.concatenate((arrs["Hit_Phi_Pho1"], arrs["Hit_Phi_Pho2"]), axis=1)
        HitsiEta = ak.concatenate((arrs["Hit_iEta_Pho1"], arrs["Hit_iEta_Pho2"]), axis=1)
        HitsiPhi = ak.concatenate((arrs["Hit_iPhi_Pho1"], arrs["Hit_iPhi_Pho2"]), axis=1)
        HitsEn = ak.concatenate((arrs["RecHitEnPho1"], arrs["RecHitEnPho2"]), axis=1)
        HitsET = ak.concatenate((arrs["RecHitETPho1"], arrs["RecHitETPho2"]), axis=1)
        HitsEZ = ak.concatenate((arrs["RecHitEZPho1"], arrs["RecHitEZPho2"]), axis=1)
        HitsX = HitsX[[j > 0.1 for j in HitsEn]]
        HitsY = HitsY[[j > 0.1 for j in HitsEn]]
        HitsZ = HitsZ[[j > 0.1 for j in HitsEn]]
        HitsEta = HitsEta[[j > 0.1 for j in HitsEn]]
        HitsPhi = HitsPhi[[j > 0.1 for j in HitsEn]]
        HitsiEta = HitsiEta[[j > 0.1 for j in HitsEn]]
        HitsiPhi = HitsiPhi[[j > 0.1 for j in HitsEn]]
        HitsEn = HitsEn[[j > 0.1 for j in HitsEn]]
        #HitsEn = np.log(np.abs(HitsEn))
        HitsET = HitsET[[j > 0.1 for j in HitsEn]]
        HitsEZ = HitsEZ[[j > 0.1 for j in HitsEn]]
        Pho1flg = arrs["Hit_X_Pho1"] * 0 + 1
        Pho2flg = arrs["Hit_X_Pho2"] * 0
        
        Phoflags = ak.concatenate((Pho1flg, Pho2flg), axis=1)
        with open("%s/EEFlag_target.pickle" % (self.outfolder), "wb") as outpickle:
            pickle.dump(Phoflags, outpickle)



        #seedidx = np.argmax(HitsEn,axis=1)
        #print("Centering rechits...")
        #x,y,z,E = [],[],[],[]
        #for index in range(len(HitsX)):
        #    seed_x = HitsX[index][seedidx[index]]
        #    seed_y = HitsY[index][seedidx[index]]
        #    seed_z = HitsZ[index][seedidx[index]]

        #    x.extend([HitsX[index] - seed_x])
        #    y.extend([HitsY[index] - seed_y])
        #    z.extend([HitsZ[index] - seed_z])
        #HitsX = ak.Array(x)
        #HitsY = ak.Array(y)
        #HitsZ = ak.Array(z)
        cf = cartfeat(
        	HitsX,
        	HitsY,
        	HitsZ,
            HitsEn,
        )
        lf = localfeat( HitsEta,HitsPhi,HitsEn)

        print("\tBuilding features took %f seconds" % (time() - t0))
        t0 = time()
        result["cartfeat"] = torchify(cf)
        result["localfeat"] = torchify(lf)
        print("\tTorchifying took %f seconds" % (time() - t0))
        t0 = time()
        with open("%s/cartfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["cartfeat"], f, pickle_protocol=4)
        with open("%s/localfeat.pickle" % (self.outfolder), "wb") as f:
            torch.save(result["localfeat"], f, pickle_protocol=4)
        print("\tDumping took %f seconds" % (time() - t0))
        return result
