# Graph Neural Network 기반 추천 시스템

이 프로젝트는 Graph Neural Network(GNN)를 활용하여 추천 시스템을 구현한 모델입니다.<br>데이터 전처리, 모델 학습, 평가를 포함한 전체 파이프라인을 제공하며 GCN, GAT, GraphSAGE 모델의 특성을 비교합니다.


## 주요 기능

1. **configs**
   - 데이터, 모델, 하이퍼파라미터 등의 설정 값으로 구성 된 파일입니다.
   - 모델 아키텍처를 기준으로 파일이 구분 되어 있습니다.
   - Hyperparameter의 경우 1개 이상의 조합이 나오게 되면 자동으로 hyperparameter 최적화를 수행합니다.

2. **데이터 불러오기**
   - `dataload/dataloader.py`는 데이터를 불러오기 위한 함수로 구성 되어 있습니다. 로컬 또는 S3 storage에 저장 된 모델을 불러오는 기능을 지원합니다.
   - `core/preprocessing.py`는 `dataload/dataloader.py`에서 정의 된 함수를 호출하여 데이터를 불러옵니다.

3. **데이터 전처리**  
   - `preprocess/preprocessor.py`에는 데이터 전처리 관련 함수들이 있습니다. 
   - `core/preprocessing.py`는 `preprocess/preprocessor.py`의 함수를 호출하여 모델 학습을 위한 데이터를 전처리 합니다.

4. **모델 구현**
   - 본 프로젝트의 모델은 크게 Projection, GNN, Predict 3파트로 나누어져 있습니다.
   - `models/projection_networks.py`는 user와 item의 feature dimension을 통일하는 역할을 합니다.
   - `models/graph_neural_networks.py`는 `configs/` 정의 된 GCN, GAT, GraphSAGE와 같은 GNN 모델을 설정 합니다.
   - `models/predict_networks.py`는 최종적으로 user와 item 사이의 score를 산출합니다.
   - `models/model.py`는 위에서 설정 한 3개의 파트를 하나의 모델로 정의합니다.
   - 모델별 하이퍼파라미터는 `configs/` 디렉토리 내 각 파일에서 설정 가능합니다.

5. **학습 및 평가**  
   - `train/train.py`는 모델 학습을 위한 함수로 구성 되어 있으며 `core/train.py`에서 해당 함수를 호출하여 모델 학습을 수행합니다. 학습 된 모델은 MLFlow의 Run에 기록됩니다. 
   - 하이퍼파라미터 최적화를 수행하게 되면 각 실험들이 MLFlow Run에 child run으로 기록 됩니다.
   - `evaluate/evaluate.py`는 학습 된 모델을 평가하는 함수로 구성 되어 있습니다. `core/evaluate.py`에서 해당 함수를 호출하여 학습 된 모델을 평가합니다. 산출 된 평가 지표는 MLFlow의 Run에 기록됩니다.

6. **유틸리티**  
   - 공통적으로 사용되는 함수와 도구들은 `utils/utils.py`에서 제공합니다.
