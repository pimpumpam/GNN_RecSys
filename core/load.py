import os
import argparse

from dataload.dataloader import DataLoader
from utils.utils import load_spec_from_config


class Loader:
    def __init__(self, cfg_meta, cfg_loader):
        self.cfg_meta = cfg_meta
        self.cfg_loader = cfg_loader
        
    def run(self):
        
        # load data
        print("[INFO] 💽 Loading Dataset")
        if self.cfg_loader.source == 'local':
            print("Loading Data from Local Directory is Under Construction")
            
            train_raw_data = DataLoader.load_from_local(
                path=self.cfg_loader.train_data_dir,
                ext=self.cfg_loader.extension
            )
            
            test_raw_data = DataLoader.load_from_local(
                path=self.cfg_loader.test_data_dir,
                ext=self.cfg_loader.extension
            )
            
        elif self.cfg_loader.source == 'S3':
            
            train_raw_data = DataLoader.load_from_s3(
                access_key=self.cfg_meta.s3['ACCESS_KEY'],
                secret_key=self.cfg_meta.s3['SECRET_KEY'],
                endpoint=self.cfg_meta.s3['STORAGE_URL'],
                bucket_name=self.cfg_meta.s3['BUCKET_NAME'],
                path=self.cfg_loader.train_data_dir,
                ext=self.cfg_loader.extension
            )
            
            test_raw_data = DataLoader.load_from_s3(
                access_key=self.cfg_meta.s3['ACCESS_KEY'],
                secret_key=self.cfg_meta.s3['SECRET_KEY'],
                endpoint=self.cfg_meta.s3['STORAGE_URL'],
                bucket_name=self.cfg_meta.s3['BUCKET_NAME'],
                path=self.cfg_loader.test_data_dir,
                ext=self.cfg_loader.extension
            )
        
        train_data = train_raw_data[self.cfg_loader.feature_field]
        test_data = test_raw_data[self.cfg_loader.feature_field]
        
        # save
        train_raw_data.to_pickle(os.path.join(self.cfg_meta.static_dir, 'train_raw_data.pkl'))
        test_raw_data.to_pickle(os.path.join(self.cfg_meta.static_dir, 'test_raw_data.pkl'))
            
        train_data.to_pickle(os.path.join(self.cfg_meta.static_dir, 'train_data.pkl'))
        test_data.to_pickle(os.path.join(self.cfg_meta.static_dir, 'test_data.pkl'))
        
#         return train_data, test_data

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conv', help="Config 파이썬 파일 명. 확장자는 제외.")
    args = parser.parse_args()
    
    # configs
    (
        cfg_meta, 
        cfg_loader, 
        _, # cfg_preprocessor
        _, # cfg_model
        _, # cfg_hyp
        _  # cfg_evaluate
    ) = load_spec_from_config(args.config)
    
    # load
    loader = Loader(cfg_meta, cfg_loader)
    loader.run()