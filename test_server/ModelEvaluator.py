import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


class ModelEvaluator:
    def __init__(self, model, X_val, y_val, X_test, y_test):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_validation_data(self, rnk_value):
        # 모델로 예측
        y_pred = self.model.predict(self.X_val)

        # 특정 시점(RNK2 값)에 해당하는 데이터프레임 추출

        val_df = pd.DataFrame({'Target': self.y_val, 'pred': y_pred, 'RNK2': rnk_value})
        val_df = val_df[val_df['RNK2'] == rnk_value][['Target', 'pred']]

        # 분류 문제의 성능 지표 계산
        accuracy = accuracy_score(val_df['Target'], val_df['pred'])
        precision = precision_score(val_df['Target'], val_df['pred'])
        recall = recall_score(val_df['Target'], val_df['pred'])
        f1 = f1_score(val_df['Target'], val_df['pred'])
        confusion = confusion_matrix(val_df['Target'], val_df['pred'])

        # 결과 출력
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print('Confusion Matrix:')
        print(confusion)

        self.save_results_to_csv(accuracy, precision, recall, f1, 'validation_results.csv')

    def evaluate_test_data(self, rnk_value):
        # 모델로 예측
        y_pred = self.model.predict(self.X_test)

        # 특정 시점(RNK2 값)에 해당하는 데이터프레임 추출
        val_df = pd.DataFrame({'Target': self.y_test, 'pred': y_pred, 'RNK2': rnk_value})
        val_df = val_df[val_df['RNK2'] == rnk_value][['Target', 'pred']]

        # 분류 문제의 성능 지표 계산
        accuracy = accuracy_score(val_df['Target'], val_df['pred'])
        precision = precision_score(val_df['Target'], val_df['pred'])
        recall = recall_score(val_df['Target'], val_df['pred'])
        f1 = f1_score(val_df['Target'], val_df['pred'])
        confusion = confusion_matrix(val_df['Target'], val_df['pred'])

        # 결과 출력
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print('Confusion Matrix:')
        print(confusion)

        self.save_results_to_csv(accuracy, precision, recall, f1, 'test_results.csv')

    def save_results_to_csv(self, accuracy, precision, recall, f1, filename):
        results_df = pd.DataFrame({
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        })

        results_df.to_csv(filename, index=False)