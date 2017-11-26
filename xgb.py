import xgboost as xgb
import pickle
from metric import get_data
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = get_data()
  
  dtrain = xgb.DMatrix(training_samples, training_labels)
  dval = xgb.DMatrix(validation_samples)
  dtest = xgb.DMatrix(test_samples)
  
  max_depths = [16, 32, 64, 128, 256, 512]
  num_rounds = [16, 32, 64, 128, 256, 512]
  learning_rates = [0.8, 1]
  
  maxFScore = 0;
  maxMax_depth = 0;
  maxNum_round = 0;
  maxLearning_rate = -1;
  
  for max_depth in max_depths:
    for num_round in num_rounds:
      for learning_rate in learning_rates:
        print 'Max depth: ' + str(max_depth) + ' Num round: ' + str(num_round) + ' Learning rate: ' + str(learning_rate)
        
        param = {'max_depth': max_depth, 'eta': learning_rate, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 24}
        bst = xgb.train(param, dtrain, num_round)
        # bst.save_model('xgboost.model')
        bst = xgb.Booster({'nthread': 4})
        # bst.load_model('xgboost.model')
        
        print 'validation set'
        validation_predictions = bst.predict(dval)
        print validation_labels
        print validation_predictions
        
        values = precision_recall_fscore_support(validation_labels, validation_predictions, average='micro')
        print values;
        f_score = values[2];
        
        if(f_score > maxFScore):
            maxFScore = f_score;
            maxMax_depth = max_depth;
            maxNum_round = num_round;
            maxLearning_rate = learning_rate;
            saved_model_filename = 'xgb_model.sav'
            pickle.dump(bst, open(saved_model_filename, 'wb'))
            
    
  loaded_model = pickle.load(open(saved_model_filename, 'rb')) 
    
  print 'test set'
  print 'Max depth: ' + str(max_depth) + ' Num round: ' + str(num_round) + ' Learning rate: ' + str(learning_rate)
  test_predictions = bst.predict(dtest)
  print test_labels
  print test_predictions
    
  print precision_recall_fscore_support(test_labels, test_predictions, average='micro')