import dgl
import collections
import pandas as pd
from itertools import product
from typing import Sequence, cast


PROGRESS_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def search_value_by_key_from_dictionary(dictionary, key):
    """
    
    """
    if key in dictionary:
        return dictionary[key]
    
    for k, v in dictionary.items():
        if isinstance(v, dict):
            item = search_value_by_key_from_dictionary(v, key)
            
            if item is not None:
                
                return item
            

def load_spec_from_config(cfg_name):
    """
    
    """
    
    meta_spec = __import__(
        f"configs.{cfg_name}", fromlist=cast(Sequence[str], [None])
    ).CfgMeta
    
    loader_spec = __import__(
        f"configs.{cfg_name}", fromlist=cast(Sequence[str], [None])
    ).CfgLoader
        
    preprocessor_spec = __import__(
        f"configs.{cfg_name}", fromlist=cast(Sequence[str], [None])
    ).CfgPreprocessor
    
    model_spec = __import__(
        f"configs.{cfg_name}", fromlist=cast(Sequence[str], [None])
    ).CfgModel
    
    hyp_spec = __import__(
        f"configs.{cfg_name}", fromlist=cast(Sequence[str], [None])
    ).CfgHyperParameter
    
    evaluate_spec = __import__(
        f"configs.{cfg_name}", fromlist=cast(Sequence[str], [None])
    ).CfgEvaluate

    return meta_spec, loader_spec, preprocessor_spec, model_spec, hyp_spec, evaluate_spec


def hyperparams_combination(cfg_hyp):
    """
    
    """
    attributes = {attr: getattr(cfg_hyp, attr) for attr in dir(cfg_hyp) if not attr.startswith("__") and not callable(getattr(cfg_hyp, attr))}
    
    hyps, vals = attributes.keys(), attributes.values()
    combinations = [dict(zip(hyps, comb)) for comb in product(*vals)]
    
    return combinations


def heterograph_to_dataframe(graph):
    """
    
    parameter
    ----------
    grpah(dgl.heterograph)
    
    return
    ----------
    (pandas.DataFrame)
    
    Usage Example
    ----------
    out_df = graph_to_dataframe(train_graph)
    """
    graph_dict = collections.defaultdict(list)
    
    for etype in graph.etypes:
        if '_by' not in etype:
            src, dst = graph.edges(etype=etype)
            
            graph_dict['user_no'].append(src.to('cpu').tolist())
            graph_dict['event_name'].append([etype]*len(src))
            graph_dict['item_no'].append(dst.to('cpu').tolist())
            
    graph_dict = {key: [item for sublist in val for item in sublist] for key, val in graph_dict.items()}
    
    return pd.DataFrame(graph_dict)


def pred_heterograph_to_dataframe(origin_graph, batch_graph, model_output_dict):
    """
    
    parameter
    ----------
    origin_graph(dgl.heterograph):
    batch_graph(dgl.heterograph):
    model_output_dict(dict):
    
    return
    ----------
    (pandas.DataFrame)
    
    Usage Example
    ----------
    out_df = graph_to_dataframe(test_graph, pos_graph, pos_score)
    
    """
    
    pred_dict = collections.defaultdict(list)
    
    for key, val in model_output_dict.items():
        
        src, etype, dst = key
        origin_eid = batch_graph.edges[etype].data[dgl.EID].to('cpu')
        
        if '_by' in etype:
            dst, src = origin_graph.find_edges(etype=etype, eid=origin_eid)
            etype = etype.replace('_by', '')
        else:
            src, dst = origin_graph.find_edges(etype=etype, eid=origin_eid)
            
        pred_dict['user_no'].append(src.tolist())
        pred_dict['event_name'].append([etype]*len(src))
        pred_dict['item_no'].append(dst.tolist())
        pred_dict['pred_score'].append(val.squeeze().to('cpu').tolist())
        
    pred_dict = {key: [item for sublist in val for item in sublist] for key, val in pred_dict.items()}
    
    return pd.DataFrame(pred_dict)