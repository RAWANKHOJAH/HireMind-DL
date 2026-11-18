"""Matched CPU training to test whether label-quality interventions help.

Arms (same frozen Train, 512-token Knowledge v1, Seed42, 3 epochs, lr 1e-4):
  original : all original 4993 labels, Hams original train_run
  pseudo   : STRICT UNVERIFIED MODEL-SUGGESTED label overlay, never published
  weight   : keep ALL original labels but downweight up to 5% per-class suspect
             rows to 0.25x CE (alternate robust training hypothesis).
Test file NEVER accessed. No automatic correction claimed as ground truth.
"""
import argparse,os,sys,json,hashlib,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,"/tmp/Senior-2/evaluation/knowledge")
from knowledge import knowledge_line,features
sys.path.insert(0,"/tmp/HireMind-DL/src")
from hiremind_dl import load_config
from hiremind_dl.data import download_raw,read_raw,load_manifest,split_frame,tokenize,as_dataset
from hiremind_dl.training import train_run
from hiremind_dl.models import build_model
from hiremind_dl.evaluation import evaluate,batch_args
from sklearn.metrics import confusion_matrix

def train_weighted(cfg,ds_train,ds_val,weights,seed=42):
    torch.set_num_threads(2)
    model=build_model("M0",cfg,seed).to("cpu")
    g=torch.Generator().manual_seed(seed)
    loader=DataLoader(ds_train,batch_size=16,shuffle=True,generator=g,num_workers=0)
    total=len(loader)*cfg["training"]["epochs"]
    assert total==939
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["selected"]["M0"]["lr"],weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(opt,int(.1*total),total)
    best={"macro_f1":-1};state=None;history=[];step=0;start=time.perf_counter()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for b in loader:
            opt.zero_grad(set_to_none=True)
            logits=model(**batch_args(b,"cpu")).logits
            losses=torch.nn.functional.cross_entropy(logits,b[3],reduction="none")
            w=weights[b[4]].to(dtype=losses.dtype)
            loss=(losses*w).sum()/w.sum()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();scheduler.step();step+=1
            if step%100==0 or step==total:
                met,_,_=evaluate(model,ds_val,"cpu",16)
                history.append({"step":step,"epoch":epoch+1,"val_accuracy":met["accuracy"],"val_macro_f1":met["macro_f1"]})
                if met["macro_f1"]>best["macro_f1"]:
                    best={**met,"step":step}
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.train()
                print(f"weighted pilot step={step}/{total} macro_f1={met['macro_f1']:.3f}",flush=True)
    return {"best":best,"history":history,"updates":step,"seconds":round(time.perf_counter()-start)},state

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",required=True,choices=["original","pseudo","weight"]);a=ap.parse_args()
    assert os.getenv("HIREMIND_MAX_LEN")=="512"
    torch.set_num_threads(2)
    cfg=load_config("/tmp/HireMind-DL/configs/experiment.json")
    assert cfg["dataset"]["split_sha256"]=="2ff3926c70152e6d5f47fbf40acebbb287e3311969618a8c7a20f45861539f9b"
    raw={"train":read_raw(download_raw(cfg,"train"))}
    manifest=load_manifest(cfg)
    tr=split_frame(raw,manifest,"train",4993)
    va=split_frame(raw,manifest,"validation",1162)
    assert set(tr.job_group).isdisjoint(set(va.job_group))
    assert tr.row_uid.is_unique and va.row_uid.is_unique
    audit=pd.read_csv("/tmp/label_audit/ALL_4993_OOF_LABEL_AUDIT.csv")
    assert len(audit)==4993 and audit.row_uid.is_unique
    audit=audit.set_index("row_uid").reindex(tr.row_uid).reset_index()
    assert audit.row_uid.tolist()==tr.row_uid.tolist()
    assert np.array_equal(audit.label_id.to_numpy(),tr.label_id.to_numpy())
    assert audit["label"].tolist()==tr["label"].tolist()
    assert audit["review_status"].eq("NOT_HUMAN_REVIEWED").all()
    before=tr.label_id.to_numpy(copy=True)
    n_change=int(audit["strict_pseudo_candidate"].astype(bool).sum())
    if a.arm=="pseudo":
        tr["label_id"]=audit.strict_pseudo_label_id.to_numpy(dtype=int)
        tr["label"]=pd.Series(tr.label_id).map({0:"No Fit",1:"Potential Fit",2:"Good Fit"})
        assert np.all(tr.label_id.between(0,2))
        assert int((tr.label_id.to_numpy()!=before).sum())==n_change
    else:
        assert np.array_equal(before,tr.label_id.to_numpy())
    tok=AutoTokenizer.from_pretrained(cfg["student"]["name"],revision=cfg["student"]["revision"])
    frames={"train":tr.copy(),"validation":va.copy()}
    tensors={}
    for part,frame in frames.items():
        inputframe=frame.copy()
        inputframe["job_description_text"]=[
           knowledge_line(features(cv,jd))+"\n"+jd
           for cv,jd in zip(frame.resume_text,frame.job_description_text)]
        tensors[part]=as_dataset(tokenize(inputframe,tok,512))
    if a.arm=="weight":
        wframe=pd.read_csv("/tmp/label_audit/EXPERIMENTAL_DOWNWEIGHT.csv")
        assert len(wframe)==4993 and wframe.row_uid.is_unique
        assert set(wframe.row_uid)==set(tr.row_uid)
        wframe=wframe.set_index("row_uid").reindex(tr.row_uid)
        assert wframe.experimental_train_weight.notna().all()
        wt=torch.tensor(wframe.experimental_train_weight.to_numpy(float),dtype=torch.float32)
        assert len(wt)==4993 and wt.min()>=.25 and wt.max()<=1
        info,state=train_weighted(cfg,tensors["train"],tensors["validation"],wt,42)
    else:
        info,state=train_run("M0",cfg["selected"]["M0"]["lr"],42,cfg,tensors["train"],tensors["validation"],"cpu")
    assert state is not None
    # Independent recomputation from saved best checkpoint
    model=build_model("M0",cfg,42)
    model.load_state_dict(state)
    metrics,logits,pred=evaluate(model,tensors["validation"],"cpu",32)
    assert abs(metrics["macro_f1"]-info["best"]["macro_f1"])<1e-7
    assert abs(metrics["accuracy"]-info["best"]["accuracy"])<1e-7
    P=logits.softmax(-1).numpy()
    out=Path("model_results");out.mkdir(exist_ok=True)
    p=va[["row_uid","job_group","label_id"]].copy()
    p=p.rename(columns={"label_id":"label"})
    p["pred"]=pred;p["p_no"]=P[:,0];p["p_potential"]=P[:,1];p["p_good"]=P[:,2]
    assert len(p)==1162 and p.row_uid.is_unique
    p.to_csv(out/f"val_predictions_{a.arm}.csv",index=False)
    wpath=out/f"weights_{a.arm}.pt"
    torch.save(state,wpath)
    digest=hashlib.sha256(wpath.read_bytes()).hexdigest()
    b=info["best"]
    result={
      "arm":a.arm,"seed":42,"train_rows":4993,"validation_rows":1162,
      "original_labels_changed":0 if a.arm!="pseudo" else n_change,
      "experimental_pseudo_label_changes":n_change if a.arm=="pseudo" else 0,
      "downweighted":int((wt<1).sum().item()) if a.arm=="weight" else 0,
      "all_original_labels_human_approved":False,
      "test_accessed":False,
      "accuracy":b["accuracy"],"macro_f1":b["macro_f1"],
      "good_fit_recall":b["good_fit_recall"],
      "good_fit_precision":100*b["per_class"]["Good Fit"]["precision"],
      "potential_fit_recall":100*b["per_class"]["Potential Fit"]["recall"],
      "confusion":b["confusion"],"best_step":b["step"],
      "updates":info["updates"],
      "checkpoint_sha256":digest,
      "source_senior_commit":"cdab258a201141706d572ba03a4a7b9fb60708db",
      "source_dl_commit":"1f22f34462c400f884bb0a00f061dbea0bd687d5",
      "warning":"Pseudo changes are NOT human-verified corrections; research ablation only"}
    (out/f"metrics_{a.arm}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
