from metric import get_data
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels = get_data()
  CArray=[0.1,1,10,100,1000]
  for C in CArray:
    print "C: " + str(C)
    clf = svm.LinearSVC(C=C)
    clf.fit(training_samples, training_labels)
    predictions = clf.predict(validation_samples)
    print precision_recall_fscore_support(validation_labels, predictions)
