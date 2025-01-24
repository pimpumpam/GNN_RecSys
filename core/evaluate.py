import os
import pickle
import argparse
import collections
import pandas as pd

import torch

import mlflow

from airflow.models import Variable

from preprocess.preprocessor import MultiColumnLabelEncoder, edge_sampler
from evaluate.evaluate import evaluator
from evaluate.metrics import precision_k, recall_k
from utils.utils import load_spec_from_config


class Evaluator:
    def __init__(self, cfg_meta, cfg_preprocessor, cfg_model, cfg_evaluate):
        self.cfg_meta = cfg_meta
        self.cfg_preprocessor = cfg_preprocessor
        self.cfg_model = cfg_model
        self.cfg_evaluate = cfg_evaluate
        
    def run(self, **kwargs):
        """
        학습 된 모델 평가 수행
        
        parameter
        ----------
        test_dataset(dgl.heterograph): Test 데이터 셋
        
        return
        ----------
        None
        
        """
        
#         RUN_ID = Variable.get("run_id", default_var=None)
        RUN_ID = kwargs['ti'].xcom_pull(key='run_id', task_ids='train_model')
        test_dataset = torch.load(os.path.join(self.cfg_meta.static_dir, 'test_graph.dgl'))
        
        print("[INFO] 🔬 Initialize Evaluate GNN Model")
        with mlflow.start_run(run_id=RUN_ID) as run:
            
            hyp = mlflow.get_run(RUN_ID).data.params
            
            sampler = edge_sampler(
                num_layers=self.cfg_model.gnn_layer['num_layers'],
                negative_sampling=True,
                num_neg_samples=int(hyp['num_negative_samples'])
            )
            print("[INFO]\t 1️⃣ Load Trained Model from MLFlow Artifacts")
            model = mlflow.pytorch.load_model(
                model_uri=os.path.join(
                    self.cfg_meta.s3['ARTIFACT_DIR'],
                    RUN_ID,
                    'artifacts',
                    self.cfg_meta.mlflow['ARTIFACT_DIR']
                )
            )
            
            print("[INFO]\t 2️⃣ Evaluate Trained Link Prediction Model")
            pred = evaluator(
                dataset=test_dataset,
                graph_sampler=sampler,
                model=model,
                batch_size=int(hyp['batch_size']),
                device=self.cfg_model.device
            )

            
            print("[INFO]\t 3️⃣ Evaluate Trained Link Prediction Model")
            
            # encoder
            encoder = MultiColumnLabelEncoder()
            with open(os.path.join(self.cfg_meta.static_dir, 'LabelEncoder.pkl'), 'rb') as r:
                encoder_dict = pickle.load(r)
            
            encoder.encoder_dict = encoder_dict
            
            encoder.inverse_transform(
                pred,
                self.cfg_preprocessor.graph_property['node']['user']['key']+self.cfg_preprocessor.graph_property['node']['item']['key'],
                inplace=True
            )
            
            # data
            true = pd.read_pickle(os.path.join(self.cfg_meta.static_dir, 'train_raw_data.pkl'))
            dtypes = {col: dtype for col, dtype in self.cfg_preprocessor.column_property['dtypes'].items() if col in pred.columns}
            pred = pred.astype(dtypes)
            pred = pred.loc[pred['pred_score']>self.cfg_evaluate.score_threshold].merge(
                true[self.cfg_preprocessor.graph_property['node']['item']['key']+self.cfg_evaluate.evaluate_feature].drop_duplicates(self.cfg_preprocessor.graph_property['node']['item']['key']),
                how='left',
                left_on=self.cfg_preprocessor.graph_property['node']['item']['key'],
                right_on=self.cfg_preprocessor.graph_property['node']['item']['key']
            )
            
            # metric            
            for step, user in enumerate(pred[self.cfg_preprocessor.graph_property['node']['user']['key'][0]].unique()):
                
                pred_ = pred.loc[pred[self.cfg_preprocessor.graph_property['node']['user']['key'][0]]==user, self.cfg_evaluate.evaluate_feature]
                true_ = true.loc[true[self.cfg_preprocessor.graph_property['node']['user']['key'][0]]==user, self.cfg_evaluate.evaluate_feature]
                
                precision = precision_k(pred_, true_, top_k=10)
#                 recall = recall_k(pred_, true_)
                
                mlflow.log_metrics(
                    {
                        'Precision_K': precision,
#                         'Recall_K': recall
                    },
                    step=step
                )
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conv', help="Config 파이썬 파일 명. 확장자는 제외.")
    args = parser.parse_args()
    
    # configs
    (
        cfg_meta, 
        _, # cfg_loader 
        cfg_preprocessor,
        cfg_model,
        _, #cfg_hyp
        cfg_evaluate
    ) = load_spec_from_config(args.config)
    
    # evaluate
    tester = Evaluator(cfg_meta, cfg_preprocessor, cfg_model, cfg_evaluate)
    tester.run()
            
            
            
            
            

        
        