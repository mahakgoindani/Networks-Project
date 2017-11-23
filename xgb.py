import xgboost as xgb
from metric import get_data
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = get_data()
  dtrain = xgb.DMatrix(training_samples, training_labels)
  dtest = xgb.DMatrix(validation_samples)
  # dtest = xgb.DMatrix(test_samples)
  max_depths = [16, 32, 64, 128, 256, 512]
  num_rounds = [16, 32, 64, 128, 256, 512]
  learning_rates = [0.8, 1]
  for max_depth in max_depths:
    for num_round in num_rounds:
      for learning_rate in learning_rates:
        print 'Max depth: ' + str(max_depth) + ' Num round: ' + str(num_round) + ' Learning rate: ' + str(learning_rate)
        param = {'max_depth': max_depth, 'eta': learning_rate, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 24}
        bst = xgb.train(param, dtrain, num_round)
        # bst.save_model('xgboost.model')
        bst = xgb.Booster({'nthread': 4})
        # bst.load_model('xgboost.model')
        predictions = bst.predict(dtest)
        print predictions
        print precision_recall_fscore_support(validation_labels, predictions, average='micro')