"""Selected 8 integration/clustering metrics + 4 label-transfer metrics, on retained cells.
INT: silhouette(cell-type ASW), calinski_harabasz, nmi, ari, isolated_label_f1, graph_connectivity, ilisi, modality_asw.
LT : accuracy, balanced_accuracy, macro_f1, weighted_f1. (calinski_harabasz normalized at plot time.)"""
import sys,numpy as np,pandas as pd,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/g/data/yr31/mp8713/project_data_integration/3_data_integration")
import evaluate_integration_metrics as EIM
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (normalized_mutual_info_score as NMI, adjusted_rand_score as ARI,
    calinski_harabasz_score, balanced_accuracy_score, f1_score)
from sklearn.neighbors import KNeighborsClassifier
INT_METRICS=["silhouette","calinski_harabasz","nmi","ari","isolated_label_f1","graph_connectivity","ilisi","modality_asw"]
LT_METRICS=["accuracy","balanced_accuracy","macro_f1","weighted_f1"]
def integ(Z,ct,mod):
    Z=np.asarray(Z,float); ct=np.asarray(ct).astype(str); mod=np.asarray(mod).astype(str); n=len(Z); k=len(set(ct.tolist()))
    o={m:np.nan for m in INT_METRICS}
    if n<5 or k<2: return o
    cl=MiniBatchKMeans(k,n_init=3,batch_size=1024,random_state=0).fit_predict(Z).astype(str)
    o["nmi"]=float(NMI(ct,cl)); o["ari"]=float(ARI(ct,cl))
    o["silhouette"]=EIM.asw_label(Z,ct)                         # (mean_sil+1)/2 -> 0..1
    try: o["calinski_harabasz"]=float(calinski_harabasz_score(Z,ct))
    except: pass
    try: o["isolated_label_f1"]=float(EIM.isolated_label_f1(ct,cl,mod))
    except: pass
    try: o["graph_connectivity"]=float(EIM.graph_connectivity(Z,ct))
    except: pass
    try: o["ilisi"]=float(EIM.lisi_scores(Z,mod,conserve=False))  # modality mixing (higher=better)
    except: pass
    try: o["modality_asw"]=float(EIM.batch_asw(Z,mod,ct))
    except: pass
    return o
def lt(Z,mod,ct):
    Z=np.asarray(Z,float); mod=np.asarray(mod).astype(str); ct=np.asarray(ct).astype(str); o={m:np.nan for m in LT_METRICS}
    i1=np.where(mod=="mod1")[0]; i2=np.where(mod=="mod2")[0]
    if len(i1)<5 or len(i2)<2: return o
    pred=KNeighborsClassifier(min(15,len(i1))).fit(Z[i1],ct[i1]).predict(Z[i2]); true=ct[i2]
    keep=np.array([x in set(ct[i1]) for x in true]); true,pred=true[keep],pred[keep]
    if not len(true): return o
    o["accuracy"]=float((true==pred).mean()); o["balanced_accuracy"]=float(balanced_accuracy_score(true,pred))
    o["macro_f1"]=float(f1_score(true,pred,average="macro",zero_division=0)); o["weighted_f1"]=float(f1_score(true,pred,average="weighted",zero_division=0))
    return o
def all_metrics(Z,ct,mod):
    o=integ(Z,ct,mod); o.update(lt(Z,mod,ct)); return o
