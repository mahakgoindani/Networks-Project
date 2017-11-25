from metric import get_data
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = get_data()
  CArray=[0.01,0.1,1,10,100,1000,10000]
  for C in CArray:
    print "C: " + str(C)
    clf = linear_model.LogisticRegression(C = C,penalty='l2')
    clf.fit(training_samples, training_labels)
    predictions = clf.predict(validation_samples)
    print precision_recall_fscore_support(validation_labels, predictions, average='micro')