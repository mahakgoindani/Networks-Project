from metrics import get_data
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels = get_data()
  
  clf = SVC()
  clf.fit(training_samples, training_labels);
  predictions = clf.predict(validation_samples);
  print precision_recall_fscore_support(validation_labels, predictions)
