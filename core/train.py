import os
import argparse

import torch
import torch.optim as optim

import mlflow

from models.model import Model
from train.train import trainer
from train.criterion import MaxMarginLoss
from preprocess.preprocessor import edge_sampler
from utils.utils import hyperparams_combination, heterograph_to_dataframe, load_spec_from_config


class Trainer:
    def __init__(self, cfg_meta, cfg_model, cfg_hyp):    
        self.cfg_meta = cfg_meta
        self.cfg_model = cfg_model
        self.cfg_hyp = cfg_hyp
        
        
    def run(self):

        print("[INFO] 🏁 Initialize Training GNN Model")
        train_dataset = torch.load(os.path.join(self.cfg_meta.static_dir, 'train_graph.dgl'))
                
        # mlflow experiment
        try:
            mlflow.create_experiment(
                self.cfg_meta.exp_name,
                artifact_location=self.cfg_meta.mlflow['ARTIFACT_DIR']
            )
            print(f"[INFO]\t 1️⃣ Experiment {self.cfg_meta.exp_name} is not exist. Create experiment.")
        except:
            print(f"[INFO]\t 1️⃣ Experienmt {self.cfg_meta.exp_name} is already exist. Execute run on the \"{self.cfg_meta.exp_name}\".")
            
        mlflow.set_experiment(self.cfg_meta.exp_name)
        
        # mlflow runs
        with mlflow.start_run(run_name=self.cfg_model.name) as run:
            
            hyp_list = hyperparams_combination(self.cfg_hyp)

            if len(hyp_list)>1:
                print("Optimal Hyperparmeter 탐색 과정 코드 추가 개발 필요")
                for hyp in hyp_list:
                    sampler = edge_sampler(
                        num_layers=self.cfg_model.gnn_layer['num_layers'],
                        negative_sampling=True, 
                        num_neg_samples=hyp['num_negative_samples']
                    )
                    
                
            elif len(hyp_list)==1:
                
                hyp = hyp_list.pop()
                
                # graph sampler
                sampler = edge_sampler(
                    num_layers=self.cfg_model.gnn_layer['num_layers'],
                    negative_sampling=True, 
                    num_neg_samples=hyp['num_negative_samples']
                )
                
                # for training
                criterion = MaxMarginLoss()
                model = Model(self.cfg_model, train_dataset.etypes)
                optimizer = optim.Adam(params=model.parameters(), lr=hyp['learning_rate'])
                
                # train
                print("[INFO]\t 2️⃣ Train Link Prediction Model")
                trainer(
                    dataset=train_dataset,
                    graph_sampler=sampler,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    batch_size=hyp['batch_size'],
                    num_epochs=hyp['num_epochs'],
                    device=self.cfg_model.device
                )
                
                # logging
                mlflow.log_params(hyp)
                mlflow.pytorch.log_model(
                    model,
                    self.cfg_meta.mlflow['ARTIFACT_DIR']
                )
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conv', help="Config 파이썬 파일 명. 확장자는 제외.")
    args = parser.parse_args()
    
    # configs
    (
        cfg_meta, 
        _, # cfg_loader 
        _, # cfg_preprocessor
        cfg_model,
        cfg_hyp, 
        _  # cfg_evaluate
    ) = load_spec_from_config(args.config)
    
    # train
    learner = Trainer(cfg_meta, cfg_model, cfg_hyp)
    learner.run()
