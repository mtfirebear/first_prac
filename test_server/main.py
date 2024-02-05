from DataProcessor import DataProcessor
from ModelEvaluator import ModelEvaluator
from XGBoostModel import XGBoostModel


def main():

    # 데이터 처리
    #data_processor = DataProcessor("C:/Users/USER/Desktop/TFT/5th_TFT_df.csv")
    data_processor = DataProcessor("/firebear_test/5th_TFT_df.csv")

    data_processor.preprocess_data()
    data_processor.split_data()

    # XGBoost 모델 학습
    xgboost_model = XGBoostModel(data_processor.X_train, data_processor.y_train,
                                  data_processor.X_val, data_processor.y_val,
                                  data_processor.X_test, data_processor.y_test)
    xgboost_model.optimize_hyperparameters()
    xgboost_model.train_model()

    # 모델 평가
    model_evaluator = ModelEvaluator(xgboost_model.model,
                                      xgboost_model.X_val, xgboost_model.y_val,
                                      xgboost_model.X_test, xgboost_model.y_test)
    model_evaluator.evaluate_validation_data(24)
    model_evaluator.evaluate_test_data(30) #6개월 후 예측
    #xgboost_model.X_test, xgboost_model.y_test, 
    #변경
if __name__ == "__main__":
    main()