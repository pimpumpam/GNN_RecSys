import dgl
import torch
from tqdm import tqdm

from utils.utils import PROGRESS_BAR_FORMAT


def trainer(dataset, graph_sampler, model, criterion, optimizer, batch_size, num_epochs, device):
    """
    모델 학습
    
    parameter
    ----------
    dataset(dgl.heterograph):
    grpah_sampler(dgl.dataloading):
    model(torch.nn):
    cretrion(object):
    optimizer(torch.optim):
    batch_size(int):
    num_epochs(int):
    device(str):
    
    return
    ----------
    None
    
    """    
    
    eids = {etype: (dataset.edges(etype=etype, form='eid')) for etype in dataset.canonical_etypes}
    dataloader = dgl.dataloading.DataLoader(dataset,
                                            eids, 
                                            graph_sampler, 
                                            device=device, 
                                            num_workers=0, 
                                            batch_size=50, # batch_size 
                                            shuffle=True, # True
                                            drop_last=True)
    
    model.to(device)
    
    print(('%20s'*3)%('Epoch', 'GPU_Mem', 'Loss'))
    for epoch in range(num_epochs):
        model.train()
        
        with tqdm(dataloader, total=len(dataloader), bar_format=PROGRESS_BAR_FORMAT) as tq:
            for step, (input_node, pos_graph, neg_graph, block) in enumerate(tq):
                
                optimizer.zero_grad()
                pos_score, neg_score = model(pos_graph, neg_graph, block)
                
                loss = criterion(pos_score, neg_score)
                loss.backward()
                optimizer.step()
                
                mem = f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.3g}G"
                tq.set_description(('%20s'*3)%(f"{epoch+1}/{num_epochs}", mem, f"{loss.item():.4}"))


