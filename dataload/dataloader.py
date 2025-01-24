import io
import os
import glob
import yaml
import boto3
import numpy as np
import pandas as pd


class DataLoader:
    
    @staticmethod
    def load_from_local(path, ext):
        
        """
        로컬에 저장 된 특정 확장자로 된 정형 데이터 불러오기

        parameter
        ----------
        path(str): 불러 올 데이터가 위치한 경로
        ext(str): 불러 올 데이터의 확장자. "csv", "excel", "feather", "pickle", "parquet" 만 가능

        return
        ----------
        data(pd.DataFrame): 특정 경로에 특정 확장자를 갖는 데이터를 모두 불러 온 결과

        """

        data = []
        loader = getattr(pd, f'read_{ext}')
        filelist = glob.glob(os.path.join(path, f'*.{ext}'))

        assert len(filelist)>0, f"\t🚨 Error occur during load data. The file with \"{ext.upper()}\" extension not exist at \"{path}\""

        for file in filelist:
            datum = loader(file)

            data.append(datum)

        data = pd.concat(data)

        print(f"\t✅ Complete loading all \"{ext.upper()}\" data. Data size: {np.shape(data)[0]} ✕ {np.shape(data)[1]}")

        return data
    
    @staticmethod
    def load_from_s3(access_key, secret_key, endpoint, bucket_name, path, ext):
        
        """
        S3 스토리지에 저장 된 특정 확장자로 된 정형 데이터 불러오기

        parameter
        ----------
        access_key(str): AWS S3 접근 키
        secret_key(str): AWS S3 보안 키
        endpoint(str): 엔드 포인트 URL
        bucket_name(str): S3 버켓 이름
        path(str): 불러 올 데이터가 위치한 경로
        ext(str): 불러 올 데이터의 확장자. "csv", "excel", "feather", "pickle", "parquet" 만 가능

        return
        ----------
        data(pd.DataFrame): 특정 경로에 특정 확장자를 갖는 데이터를 모두 불러 온 결과

        """
        
        data = []
        loader = getattr(pd, f'read_{ext}')
        
        client = boto3.client(service_name='s3',
                              aws_access_key_id=access_key,
                              aws_secret_access_key=secret_key,
                              endpoint_url=endpoint)
        
        obj_list = client.list_objects_v2(Bucket=bucket_name, Prefix=path).get("Contents")
        
        assert len(obj_list)>0, f"\t🚨 Error occur during load data. The file with \"{ext.upper()}\" extension not exist at \"{path}\""
        
        for obj in obj_list:
            
            obj_name = obj['Key']
            
            if obj_name.split('.')[-1] == ext:
                response = client.get_object(Bucket=bucket_name, Key=obj_name)
                obj_content = response['Body'].read()
                
                datum = loader(io.BytesIO(obj_content))
                
                data.append(datum)
                
        data = pd.concat(data)
        
        print(f"\t✅ Complete loading all \"{ext.upper()}\" data. Data size: {np.shape(data)[0]} ✕ {np.shape(data)[1]}")
        
        return data
