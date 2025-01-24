import os
import mlflow
import argparse
import warnings

from core.load import Loader
from core.preprocessing import Preprocessor
from core.train import Trainer
from core.evaluate import Evaluator
from utils.utils import load_spec_from_config

warnings.filterwarnings('ignore')


class Run():
    def __init__(self, config_name):
        (
            self.cfg_meta,
            self.cfg_loader,
            self.cfg_preprocessor,
            self.cfg_model,
            self.cfg_hyp,
            self.cfg_evaluate
        ) = load_spec_from_config(config_name)
        
        
    def load_data(self):
        loader = Loader(self.cfg_meta, self.cfg_loader)
        loader.run()
        
    def preprocess(self):
        preprocessor = Preprocessor(self.cfg_meta, self.cfg_preprocessor)
        preprocessor.run()
        
    def train(self):
        trainer = Trainer(self.cfg_meta, self.cfg_model, self.cfg_hyp)
        trainer.run()
        
    def evaluate(self):
        evaluator = Evaluator(self.cfg_meta, self.cfg_preprocessor, self.cfg_model, self.cfg_evaluate)
        evaluator.run()
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='conv', help="Config 파이썬 파일명. 확장자 제외.")
    args = parser.parse_args()
    
    executor = Run(args.config)
    
    # load data
    executor.load_data()
    
    # preprocessing data
    executor.preprocess()
    
    # train model
    executor.train()
    
    # evaluate model
    executor.evaluate()
    
