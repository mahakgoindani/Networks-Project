from metric import get_data
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import pickle

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = get_data()
  CArray=[0.1,1,10,100,1000]
  
  maxFScore = 0;
  maxC = 0;
  
  for C in CArray:
    print "C: " + str(C)
    clf = svm.LinearSVC(C=C)
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
        maxC = C;
        saved_model_filename = 'model.sav'
        pickle.dump(clf, open(saved_model_filename, 'wb'))
            
   #load the model
  loaded_model = pickle.load(open(saved_model_filename, 'rb')) 
    
  print 'test set'
  test_predictions = clf.predict(test_samples)
  print test_labels
  print test_predictions
    
  print precision_recall_fscore_support(test_labels, test_predictions, average='micro')
