# GNN_RecSys

---

이 프로젝트는 Graph Neural Network(GNN)를 활용하여 추천 시스템을 구현한 모델입니다. 다양한 GNN 모델(GCN, GAT, GraphSAGE 등)을 실험하고, 데이터 전처리, 모델 학습, 평가를 포함한 전체 파이프라인을 제공합니다.


## 디렉토리 구조
configs
core
dataload
datasets
evaluate
models
preprocess
static
train
utils
run.py


## 주요 기능

1. **데이터 전처리**  
   - `preprocess/preprocessor.py`와 `core/preprocessing.py`에서 데이터 준비와 전처리를 수행합니다.

2. **모델 구현**  
   - `models/graph_neural_networks.py`에서 GCN, GAT, GraphSAGE 등 다양한 GNN 모델을 정의합니다.
   - 모델별 하이퍼파라미터는 `configs/` 디렉토리에서 설정 가능합니다.

3. **학습 및 평가**  
   - `train/train.py`에서 모델 학습을 수행하며, 평가 지표는 `evaluate/metrics.py`에서 제공합니다.

4. **유틸리티**  
   - 공통적으로 사용되는 함수와 도구들은 `utils/utils.py`에서 제공합니다.
