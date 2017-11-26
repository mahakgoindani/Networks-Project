from metric import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = get_data()
  
  max_depths = [5,7,9,11,13,15]
  num_trees = [5,7,9,11,13,15]
  
  maxFScore = 0;
  maxMax_depth = 0;
  maxNum_tree = 0;
  
  for max_depth in max_depths:
    for num_tree in num_trees:
      print "Max Depth: " + str(max_depth) + " Num Trees: " + str(num_tree)
      
      clf = RandomForestClassifier(n_estimators=num_tree, max_depth=max_depth)
      clf.fit(training_samples, training_labels)
      
      print 'validation set'
      validation_predictions = clf.predict(validation_samples)
      print validation_labels
      print validation_predictions
      
      values = precision_recall_fscore_support(validation_labels, validation_predictions, average='micro')
      print values;
      f_score = values[2];
    
      if(f_score > maxFScore):
        maxFScore = f_score;
        maxMax_depth = max_depth;
        maxNum_tree = num_tree;
        saved_model_filename = 'rf_model.sav'
        pickle.dump(clf, open(saved_model_filename, 'wb'))
      
  loaded_model = pickle.load(open(saved_model_filename, 'rb')) 
    
  print 'test set'
  print "Max Depth: " + str(maxMax_depth) + " Num Trees: " + str(maxNum_tree)
  test_predictions = clf.predict(test_samples)
  print test_labels
  print test_predictions
    
  print precision_recall_fscore_support(test_labels, test_predictions, average='micro')    
