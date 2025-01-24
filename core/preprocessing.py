import os
import argparse
import pandas as pd

import torch

from preprocess.preprocessor import MultiColumnLabelEncoder, MultiColumnScaler, tabular_to_heteroGraph
from utils.utils import load_spec_from_config, search_value_by_key_from_dictionary

    
class Preprocessor:
    
    def __init__(self, cfg_meta, cfg_preprocessor):
        
        self.cfg_meta = cfg_meta
        self.column_property = cfg_preprocessor.column_property
        self.graph_property = cfg_preprocessor.graph_property
        self.scaler_property = cfg_preprocessor.scaler_property
        
        
    def run(self):
        # load data
        train_data = pd.read_pickle(os.path.join(self.cfg_meta.static_dir, 'train_data.pkl'))
        test_data = pd.read_pickle(os.path.join(self.cfg_meta.static_dir, 'test_data.pkl'))
        
        # objects
        train_encoder = MultiColumnLabelEncoder() 
        test_encoder = MultiColumnLabelEncoder()
        scaler = MultiColumnScaler(self.scaler_property['scaler'])
        
        
        # encoding
        print("[INFO] 🛠️ Preprocessing Dataset")
        print("[INFO]\t 1️⃣ Encoding for Nominal Fields")
        train_encoder.fit_transform(
            df=train_data,
            columns=self.column_property['nominal_field'],
            inplace=True,
            save_pkl=True,
            save_path=self.cfg_meta.static_dir
        )
        
        train_encoder.transform(
            df=test_data,
            columns=self.column_property['nominal_field'],
            inplace=True
        )
        
        test_encoder.fit_transform(
            df=test_data,
            columns=self.graph_property['node']['user']['key']+self.graph_property['node']['item']['key'],
            inplace=True,
#             save_pkl=True, 
#             save_path=self.cfg_meta.static_dir
        )
        
        
        # scaling
        print("[INFO]\t 2️⃣ Scaling for Numeric Fields")
        scaler.fit_transform(
            train_data,
            self.column_property['numeric_field'],
            inplace=True,
            save_pkl=True,
            save_path=self.cfg_meta.static_dir
        )
        
        scaler.transform(
            test_data,
            self.column_property['numeric_field'],
            inplace=True
        )
        
        
        
        print("[INFO]\t 3️⃣ Transform Dataframe to Graph")
        train_data[self.graph_property['edge']['event']['key'][0]] = train_data[self.graph_property['edge']['event']['key'][0]].map({'click_item': 'click',
                                                                                               'like_item': 'like',
                                                                                               'add_to_cart': 'cart',
                                                                                               'purchase_success': 'buy'})
        
        test_data[self.graph_property['edge']['event']['key'][0]] = test_data[self.graph_property['edge']['event']['key'][0]].map({'click_item': 'click',
                                                                                             'like_item': 'like',
                                                                                             'add_to_cart': 'cart',
                                                                                             'purchase_success': 'buy'})
        
        train_graph = tabular_to_heteroGraph(train_data, self.graph_property)
        test_graph = tabular_to_heteroGraph(test_data, self.graph_property)
        
        # save
        torch.save(
            train_graph,
            os.path.join(self.cfg_meta.static_dir, 'train_graph.dgl')
        )
        torch.save(
            test_graph,
            os.path.join(self.cfg_meta.static_dir, 'test_graph.dgl')
        )
        
#         return train_graph, test_graph

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conv', help="Config 파이썬 파일 명. 확장자는 제외.")
    args = parser.parse_args()
    
    # configs
    (
        cfg_meta, 
        _, # cfg_loader 
        cfg_preprocessor, 
        _, # cfg_model
        _, # cfg_hyp
        _  # cfg_evaluate
    ) = load_spec_from_config(args.config)
    
    # preprocessing
    preprocessor = Preprocessor(cfg_meta, cfg_preprocessor)
    preprocessor.run()
