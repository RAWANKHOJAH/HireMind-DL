"""HireMind 4,993-label OOF audit. This is screening, NOT adjudication.

Each candidate is predicted by a BERT-Tiny NOT trained on its job group or
itself. All five models fit 3 full epochs on other 4 folds. The heldout fold
is NEVER used for checkpoint selection, thresholds, or early stopping.
Use Hams original frozen split and original Knowledge v1. NO test access.
No raw CV/job text written to any artifact.
"""
import os,sys,json,hashlib,time,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader,Subset
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,"/tmp/Senior-2/evaluation/knowledge")
from knowledge import knowledge_line,features
sys.path.insert(0,"/tmp/HireMind-DL/src")
from hiremind_dl import load_config
from hiremind_dl.data import download_raw,read_raw,load_manifest,split_frame,tokenize,as_dataset
from hiremind_dl.models import build_model,seed_everything
from hiremind_dl.evaluation import batch_args

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fold",type=int,required=True,choices=range(5));a=ap.parse_args()
    assert os.environ.get("HIREMIND_MAX_LEN")=="512"
    cfg=load_config("/tmp/HireMind-DL/configs/experiment.json")
    assert cfg["dataset"]["split_sha256"]=="2ff3926c70152e6d5f47fbf40acebbb287e3311969618a8c7a20f45861539f9b"
    raw={"train":read_raw(download_raw(cfg,"train"))}
    manifest=load_manifest(cfg)
    d=split_frame(raw,manifest,"train",4993)
    assert d.row_uid.is_unique and d.label_id.nunique()==3
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261118)
    foldspec=list(splitter.split(d,d.label_id,groups=d.job_group))
    fit,held=foldspec[a.fold]
    assert set(d.job_group.iloc[fit]).isdisjoint(d.job_group.iloc[held])
    assert len(set(fit)&set(held))==0
    assert sorted([x for _,h in foldspec for x in h])==list(range(len(d)))
    tok=AutoTokenizer.from_pretrained(cfg["student"]["name"],revision=cfg["student"]["revision"])
    aug=d.copy()
    aug["job_description_text"]=[knowledge_line(features(cv,job))+"\n"+job
                                 for cv,job in zip(d.resume_text,d.job_description_text)]
    ds=as_dataset(tokenize(aug,tok,512))
    seed=42+a.fold
    torch.set_num_threads(2)
    model=build_model("M0",cfg,seed).to("cpu")
    g=torch.Generator().manual_seed(seed)
    train_ds=Subset(ds,fit.tolist())
    val_ds=Subset(ds,held.tolist())
    train_loader=DataLoader(train_ds,batch_size=16,shuffle=True,generator=g,num_workers=0)
    total=3*len(train_loader)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["selected"]["M0"]["lr"],weight_decay=0.01)
    sched=get_linear_schedule_with_warmup(optimizer,int(0.1*total),total)
    start=time.time()
    for epoch in range(3):
        model.train();loss_sum=0
        for step,b in enumerate(train_loader,1):
            optimizer.zero_grad(set_to_none=True)
            loss=model(**batch_args(b,"cpu")).loss
            assert torch.isfinite(loss).item(),f"nonfinite loss in fold {a.fold}"
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();sched.step()
            loss_sum+=float(loss.item())
            if step%100==0:
                print(f"fold={a.fold} epoch={epoch+1} step={step}/{len(train_loader)} loss={loss_sum/step:.4f}",flush=True)
        print(f"fold={a.fold} epoch={epoch+1} mean_loss={loss_sum/len(train_loader):.5f}",flush=True)
    model.eval();logits=[];labels=[]
    with torch.no_grad():
        for b in DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=0):
            output=model(**batch_args(b,"cpu"))
            logits.append(output.logits.cpu());labels.extend(b[3].tolist())
    probs=torch.cat(logits).softmax(-1).numpy()
    assert len(probs)==len(held) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(1),1,atol=2e-5)
    assert list(labels)==d.label_id.iloc[held].tolist()
    pred=probs.argmax(axis=1)
    out=Path("oof_fold");out.mkdir(exist_ok=True)
    res=d.iloc[held][["row_uid","cv_id","job_id","job_group","label","label_id"]].copy().reset_index(drop=True)
    res["fold"]=a.fold
    res["p_no"]=probs[:,0]
    res["p_potential"]=probs[:,1]
    res["p_good"]=probs[:,2]
    res["oof_prediction"]=pred
    res.to_csv(out/f"oof_fold_{a.fold}.csv",index=False)
    meta={
     "fold":a.fold,"seed":seed,"train_rows":len(fit),"heldout_rows":len(held),
     "train_job_groups":int(d.job_group.iloc[fit].nunique()),
     "heldout_job_groups":int(d.job_group.iloc[held].nunique()),
     "training_epochs":3,"train_updates":total,"no_validation_checkpoint_selection":True,
     "group_leakage":0,"heldout_accuracy_against_given_labels":float(accuracy_score(labels,pred)),
     "heldout_macro_f1_against_given_labels":float(f1_score(labels,pred,average="macro",zero_division=0)),
     "time_sec":round(time.time()-start),
     "source_senior_commit":"cdab258a201141706d572ba03a4a7b9fb60708db",
     "source_DL_commit":"1f22f34462c400f884bb0a00f061dbea0bd687d5",
     "test_accessed":False}
    (out/f"oof_fold_{a.fold}_meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=="__main__":main()
