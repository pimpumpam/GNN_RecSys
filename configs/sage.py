class CfgMeta:
    name = 'Ecommerce-GCN'
    exp_name = 'dgl_ecommerce'
    static_dir = '/서버내프로젝트경로/static'
    sqlite = {
        'DATABASE_DIR': '/서버내프로젝트경로/mlflow.db'
    }
    
    mlflow = {
        'DASHBOARD_URL': '서버URL:포트번호',
        'ARTIFACT_DIR': 'ecommerce_gnn'
    }
    
    s3 = {
        'STORAGE_URL': '서버URL:포트번호',
        'ARTIFACT_DIR': 's3://버킷이름/개인폴더경로',
        'ACCESS_KEY': '보안접근키',
        'SECRET_KEY': '보안비밀키',
        'BUCKET_NAME': '버킷이름'
    }
    
class CfgLoader:
    source = 'S3' # S3, local
    extension = 'feather'
    train_data_dir = '/프로젝트경로/GNN_RecSys/datasets/train'
    test_data_dir = '/프로젝트경로/GNN_RecSys/datasets/train'
    feature_field = ['event_name',
                     'user_no', 'gender', 'age', 'mobile_brand_name', 'region',
                     'item_no', 'price', 'brand_no', 'category_vector']
    
    
class CfgPreprocessor:
    
    column_property = {
        'numeric_field': ['age', 'price'],
        'nominal_field': ['user_no', 'item_no', 'gender', 'brand_no', 'region', 'mobile_brand_name'],
        'dtypes': {
            'event_name': str,
            'user_no': str,
            'item_no': str,
            'gender': str,
            'brand_no': str,
            'region': str,
            'mobile_brand_name': str,
            'age': float,
            'price': float
        }
    }
    
    graph_property = {
        'node': {
            'user': {
                'key': ['user_no'],
                'feature': ['gender', 'age','mobile_brand_name','region']
            },
            'item':{
                'key': ['item_no'],
                'feature': ['price', 'brand_no', 'category_vector']
            }
        },
        'edge': {
            'event':{
                'key': ['event_name'],
                'feature': ['click', 'like', 'cart', 'buy']
            }
        }
    }

    scaler_property = {
        'scaler': 'sklearn.preprocessing.MinMaxScaler',
    }
    
    
class CfgModel:
    name = 'Graph SAGE Networks'
    device = 'cuda:0' # cpu, cuda:0,1,2,3
    
    projection_layer = {
        'num_layers': 3,
        'architecture': {
            'user': [
                ['nn.Linear', [4, 50]], # args: [input_dimension, output_dimension]
                ['nn.ReLU', [False]],
                ['nn.Linear', [50, 100]],
                ['nn.ReLU', [False]],
                ['nn.Linear', [100, 200]]
            ],
            'item': [
                ['nn.Linear', [1602, 800]],
                ['nn.ReLU', [False]],
                ['nn.Linear', [800, 400]],
                ['nn.ReLU', [False]],
                ['nn.Linear', [400, 200]]
            ]
        }
    }
    
    gnn_layer = {
        'num_layers': 3,
        'aggregate_method': 'sum', # sum, max, min, mean, stack
        'multihead_merge_method': 'concat', # concat, mean
        'architecture': [
            ['dglnn.SAGEConv', [200, 100, 'mean', 0.0, True, 'null', 'F.relu']], # mean, pool, lstm
            ['dglnn.SAGEConv', [100, 100, 'mean', 0.0, True, 'null', 'F.relu']],
            ['dglnn.SAGEConv', [100, 50, 'mean']]
        ]
    }
    
    predict_layer = {
        'num_layers': 2,
        'predict_method': 'dot', # dot, linear
        'architecture': [
            ['nn.Linear', [50, 25]],
            ['nn.ReLU', [False]],
            ['nn.Linear', [25, 1]]
        ]
    }


class CfgHyperParameter:
    batch_size = [100]
    learning_rate = [0.0005]
    num_negative_samples = [3]
    num_epochs = [1]

    
class CfgEvaluate:
    score_threshold = -1
    evaluate_feature = ['category3_name']
    
    
    
class CfgAnalyze:
    performance_index = "precision" # precision, recall
    
