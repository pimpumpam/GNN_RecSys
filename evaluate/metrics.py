def precision_k(pred, truth, top_k):
    """
    Precision@K를 활용한 모델 평가
    
    parameter
    ----------
    pred(pandas.DataFrame): 모델 예측 결과가 포함 된 데이터프레임
    truth(pandas.DataFrame): 실제 원본 데이터
    top_k(int): 추천 리스트 중 Top N개에 대한 제한
    eval_col(str): Config 파일 내 설정 된 평가 기준이 되는 컬럼
    
    return
    ----------
    metrics(list): Top K개에 대한 precision 결과
    
    """    

    num_hit = len(set(pred).intersection(set(truth)))
    precision = float(num_hit) / top_k
    
    return precision


def recall_k(pred, truth):
    """
    Recall@K를 활용한 모델 평가
    
    parameter
    ----------
    pred(pandas.DataFrame): 모델 예측 결과가 포함 된 데이터프레임
    truth(pandas.DataFrame): 실제 원본 데이터
    top_k(int): 추천 리스트 중 Top N개에 대한 제한
    eval_col(str): Config 파일 내 설정 된 평가 기준이 되는 컬럼
    
    return
    ----------
    metrics(list): Top K개에 대한 recall 결과
    
    """
    
    num_hit = len(set(pred).intersection(set(truth)))
    recall = float(num_hit) / len(truth)
    
    return recall