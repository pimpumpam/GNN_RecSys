import dgl
import torch
import pandas as pd
from tqdm import tqdm

from utils.utils import PROGRESS_BAR_FORMAT, pred_heterograph_to_dataframe


def evaluator(dataset, graph_sampler, model, batch_size, device):
    """
    학습 된 모델 평가
    
    parameter
    ----------
    dataset(dgl.heterograph):
    grpah_sampler(dgl.dataloading):
    model(torch.nn):
    batch_size(int):
    devcie(str):
    
    return
    ----------
    pred_result(pandas.DataFrame): 
    
    """
    
    pred_result = []
    
    eids = {etype: (dataset.edges(etype=etype, form='eid')) for etype in dataset.canonical_etypes}
    dataloader = dgl.dataloading.DataLoader(dataset,
                                            eids,
                                            graph_sampler,
                                            device=device,
                                            num_workers=0,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True)
    
    model.to(device)
    model.eval()
    
    print(('%20s'*3)%('Iteration', 'GPU_Mem', ''))
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader), bar_format=PROGRESS_BAR_FORMAT) as tq:
            for step, (input_node, pos_graph, neg_graph, block) in enumerate(tq):
                
                pos_score, _ = model(pos_graph, neg_graph, block)
                
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                tq.set_description(('%20s'*3) % (f"{step+1}/{len(dataloader)}", mem, ' '))
                
                pred_df = pred_heterograph_to_dataframe(dataset, pos_graph, pos_score)
                pred_result.append(pred_df)                
                
    return pd.concat(pred_result).reset_index(drop=True)
