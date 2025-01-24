import os
import pickle
import collections
import numpy as np
from itertools import chain

import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder

import dgl
import torch


class MultiColumnLabelEncoder:
    
    def __init__(self):
        
        """
        Initializer
        
        각 Columns에 적용 될 Label Encoder를 딕셔너리 형태로 구현
        dict = {columns명 : 각 columns의 정보를 갖는 label encoder} 
        
        """
        
        self.encoder_dict = collections.defaultdict(LabelEncoder)
        
    
    def transform(self, df, columns, inplace=False):
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = df[columns].apply(lambda x: self.encoder_dict[x.name].transform(x))
            
            return df
        
        else:
            df[columns] = df[columns].apply(lambda x: self.encoder_dict[x.name].transform(x))
            
        print(f"\t\t✅ Complete transformation using LabelEncoder")
    
    
    def fit_transform(self, df, columns, inplace=False, save_pkl=False, **kwargs):
        
        """
        Argument로 주어진 multi column의 각 컬럼에 대해 Label Encoding 수행
        
        parameter
        ---------
        df(pandas.DataFrame): Label Encoding을 적용할 Column이 포함된 DataFrame
        columns(list): DataFrame 내 Label Encoding을 적용할 컬럼명
        inplace(boolean): Argument로 사용된 DataFrame 변환 유지 여부
        save_pkl(boolean): LabelEncoder를 pkl 형식으로 저장 할지에 대한 여부
        kwargs:
            save_path(str): LabelEncoder 객체 저장 경로
        
        return
        ----------
        df(pandas.DataFrame): Label Encoding이 적용 된 DataFrame
        
        """
        
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = df[columns].apply(lambda x: self.encoder_dict[x.name].fit_transform(x))
            
            return df
        
        else:
            df[columns] = df[columns].apply(lambda x: self.encoder_dict[x.name].fit_transform(x))
            
        print(f"\t\t✅ Complete fitting & transformation using LabelEncoder")
        
        
        if save_pkl:
            with open(os.path.join(kwargs['save_path'], f"LabelEncoder.pkl"), 'wb') as f:
                pickle.dump(self.encoder_dict, f)
                
            print(f"\t\t✅ Complete save \"LabelEncoder\"")
    
    
    def inverse_transform(self, df, columns, inplace=False):
        
        """
        Multi Columns에 대해 변환 된 Label Encoder 정보를 기반으로 원래 형태로 역변환
        
        parameter
        ----------
        df(pandas.DataFrame): 역변환을 적용할 Column이 포함된 DataFrame
        columns(list): 역변환을 적용할 Column 명으로 구성된 List
        inplace(boolean): Argument로 사용된 DataFrame 변환 유지 여부
        
        return
        ----------
        df(pandas.DataFrame): 역변환이 적용 된 DataFrame
        
        """
        
        if not isinstance(columns, list):
            columns = [columns]
            
        if not all(key in self.encoder_dict for key in columns):
            raise KeyError(f'One of column in {columns} is not encoded')
            
        if not inplace:
            df = df.copy()
            df[columns] = df[columns].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x))
            
            return df
        
        else:
            df[columns] = df[columns].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x))

        print(f"\t\t✅ Complete inverse transformation using LabelEncoder")

        
        
class MultiColumnScaler:
    
    def __init__(self, scaler_kind):
        
        """
        Initializer
        
        paramater
        ----------
        scaler_kind(str): 각 Column에 적용 할 Scailer 명. sklearn.preprocessing.StandardScaler/MinMaxSclaer 사용 가능
        
        """
        
        self.scaler_kind = scaler_kind
        self.scaler = eval(scaler_kind)()
        
    
    def transform(self, df, columns, inplace=False):
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = self.scaler.transform(df[columns])
            
            return df
        
        else:
            df[columns] = self.scaler.transform(df[columns])
    
        print(f"\t\t✅ Complete transformation using {type(self.scaler).__name__}")
        
    
    def fit_transform(self, df, columns, inplace=False, save_pkl=False, **kwargs):
        
        """
        Argument로 주어진 multi column의 각 컬럼에 대해 Scaler 적용
        
        parameter
        ---------
        df(pandas.DataFrame): Scaling 할 Column이 포함된 DataFrame
        columns(list): DataFrame 내 Scaler를 적용할 컬럼명
        inplace(boolean): Argument로 사용된 DataFrame 변환 유지 여부
        save_pkl(boolean): Scaler 객체를 pkl 형식으로 저장 할지에 대한 여부
        kwargs:
            save_path(str): Scaler 객체 저장 경로
        
        return
        ----------
        df(pandas.DataFrame): Label Encoding이 적용 된 DataFrame
        
        """
            
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = self.scaler.fit_transform(df[columns])
            return df
        
        else:
            df[columns] = self.scaler.fit_transform(df[columns])
            
        print(f"\t\t✅ Complete fitting & transformation using {type(self.scaler).__name__}")
        
        if save_pkl:
            with open(os.path.join(kwargs['save_path'], f"{type(self.scaler).__name__}.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print(f"\t\t✅ Complete save \"{type(self.scaler).__name__}\"")
            
            
    def inverse_transform(self, df, columns, inplace=False):
        
        """
        Scaler 정보를 기반으로 Multi Columns 정보 역변환
        
        parameter
        ----------
        df(pandas.DataFrame): 역변환을 적용할 Column이 포함된 DataFrame
        columns(list): 역변환을 적용할 Column 명으로 구성된 List
        inplace(boolean): Argument로 사용된 DataFrame 변환 유지 여부
        
        return
        ----------
        df(pandas.DataFrame): 역변환이 적용 된 DataFrame
        
        """
        
        if not isinstance(columns, list):
            columns = [columns]
            
        if not inplace:
            df = df.copy()
            df[columns] = self.scaler.inverse_transform(df[columns])
            
            return df
        
        else:
            df[columns] = self.scaler.inverse_transform(df[columns])
            
        print(f"\t\t✅ Complete inverse transformation using {type(self.scaler).__name__}")
        
        
def tabular_to_heteroGraph(data, graph_property):
    
    user_data = data[list(chain(*graph_property['node']['user'].values()))] \
                    .drop_duplicates(subset=graph_property['node']['user']['key'][0]) \
                    .sort_values(graph_property['node']['user']['key'][0]) \
                    .reset_index(drop=True)
    item_data = data[list(chain(*graph_property['node']['item'].values()))] \
                    .drop_duplicates(subset=graph_property['node']['item']['key'][0]) \
                    .sort_values(graph_property['node']['item']['key'][0]) \
                    .reset_index(drop=True)
    event_data = data[graph_property['node']['user']['key'] \
                      + graph_property['node']['item']['key'] \
                      + graph_property['edge']['event']['key']]
     
    # Convert dataframe to hetrogeneous-graph
    hetero_graph = {}
    for e in graph_property['edge']['event']['feature']:
        src = event_data.loc[event_data[graph_property['edge']['event']['key'][0]]==e, graph_property['node']['user']['key'][0]].to_numpy()
        dst = event_data.loc[event_data[graph_property['edge']['event']['key'][0]]==e, graph_property['node']['item']['key'][0]].to_numpy()

        hetero_graph[(graph_property['node']['user']['key'][0], e, graph_property['node']['item']['key'][0])] = (src, dst)
        hetero_graph[(graph_property['node']['item']['key'][0], f'{e}_by', graph_property['node']['user']['key'][0])] = (dst, src)

    hetero_graph = dgl.heterograph(hetero_graph)

    # Set node feature
    for k, v in graph_property['node'].items():
        sub_k = v['key'][0]
        sub_f = v['feature']
        
        if k == 'user':
            hetero_graph.nodes[sub_k].data['feature'] = torch.tensor(user_data[sub_f].values, dtype=torch.float32)
        elif k == 'item':
            hetero_graph.nodes[sub_k].data['feature'] = torch.cat((torch.tensor(item_data[['price', 'brand_no']].values, dtype=torch.float32),
                                                                   torch.tensor(np.array(item_data['category_vector'].values.tolist()), dtype=torch.float32)), 
                                                                  dim=1)

    return hetero_graph


def edge_sampler(num_layers, negative_sampling=False, **kwargs):
    """
    
    """
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    
    if negative_sampling:
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(kwargs['num_neg_samples'])
        
        return dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=neg_sampler)
    
    return dgl.dataloading.as_edge_prediction_sampler(sampler)
