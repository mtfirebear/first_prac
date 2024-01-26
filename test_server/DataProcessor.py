import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb
import optuna
import random
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

class DataProcessor:
    pd.set_option('mode.chained_assignment', None)  # SettingWithCopyWarning 경고를 무시

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, encoding='utf-8')

    def preprocess_data(self):
        # ID 리스트 생성
        IDs = self.df['ID'].unique()
        id_list = IDs

        # 결과를 저장할 리스트 초기화
        results = []

        # 각 ID에 대해 처리
        for id_value in id_list:
            # 해당 ID의 데이터를 선택
            id_df = self.df[self.df['ID'] == id_value]

            # 시점 t+6의 '미납일수' 값이 29 이상인지 확인
            target_value = id_df['미납일수'].shift(-6) >= 29

            # 'Target' 컬럼에 결과 추가
            id_df['Target'] = target_value

            # 결과를 결과 리스트에 추가
            results.append(id_df)

        # 결과를 하나의 데이터프레임으로 합치기
        self.data = pd.concat(results)

        # t+6의 시점의 값을 가져왔으므로 31~36은 못쓴다.
        self.data = self.data[~self.data['RNK2'].isin([31,32,33,34,35,36])]

        # 원본 리스트
        original_list = list(self.data['ID'].unique())

        # 리스트 섞기
        random.seed(42)
        random.shuffle(original_list)

        # 리스트 길이
        total_length = len(original_list)

        # 8:2 비율로 분할할 인덱스
        train_index = int(total_length * 0.8 * 0.8)
        val_index = int(total_length * 0.8)

        # 8:2 비율로 분할
        self.train_ID = original_list[:train_index]
        self.val_ID = original_list[train_index:val_index]
        self.test_ID = original_list[val_index:]

    def split_data(self):
        # 결과 출력
        print("Train 데이터:", len(self.train_ID))
        print("Validation 데이터:", len(self.val_ID))
        print("Test 데이터:", len(self.test_ID))

        self.train = self.data[self.data['ID'].isin(self.train_ID)]
        self.val = self.data[self.data['ID'].isin(self.val_ID)]
        self.test = self.data[self.data['ID'].isin(self.test_ID)]

        self.train = self.train.sort_values(by=['ID','RNK2'])
        self.val = self.val.sort_values(by=['ID','RNK2'])
        self.test = self.test.sort_values(by=['ID','RNK2'])

        self.train.reset_index(drop=True, inplace=True)
        self.val.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

        #Target에 대한 비율 확인
        print(f"Train_Target : {self.train[['Target']].value_counts()}")
        print(f"Validation_Target : {self.val[['Target']].value_counts()}")
        print(f"Test_Target : {self.test[['Target']].value_counts()}")

        continuous_column = ['연령', '소득', '대출기간', '대출금리', '부부합산연소득', '세대원수', '자녀수', '총보유부채', '상환예정원금', '상환예정이자',
                            '상환예정원리금', '회차상환원금', '회차상환이자', '회차내원리금연체지연배상금', '잔액연체지연배상금',
                            '미납일수', '회차내조기상환금액', '조기상환이자', '조기상환수수료', '회차거래횟수', '상환후잔액', 'BS스코어',
                            '신규취급액기준COFIX', '주택구입부담지수']

        # StandardScaler 객체 생성
        scaler = MinMaxScaler()

        # 데이터에 대한 정규화 수행
        self.train[continuous_column] = pd.DataFrame(scaler.fit_transform(self.train[continuous_column]))
        self.val[continuous_column] = pd.DataFrame(scaler.transform(self.val[continuous_column]))
        self.test[continuous_column] = pd.DataFrame(scaler.transform(self.test[continuous_column]))

        self.train = self.train[~self.train['RNK2'].isin([25,26,27,28,29,30])]

        self.X_train = self.train[continuous_column]
        self.y_train = self.train['Target']
        self.X_val = self.val[continuous_column]
        self.y_val = self.val['Target']
        self.X_test = self.test[continuous_column]
        self.y_test = self.test['Target']