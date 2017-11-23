from metric import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = get_data()
  max_depths = [5,7,9,11,13,15]
  num_trees = [5,7,9,11,13,15]
  for max_depth in max_depths:
    for num_tree in num_trees:
      print "Max Depth: " + str(max_depth) + " Num Trees: " + str(num_tree)
      clf = RandomForestClassifier(n_estimators=num_tree, max_depth=max_depth)
      clf.fit(training_samples, training_labels)
      predictions = clf.predict(validation_samples)
      print precision_recall_fscore_support(validation_labels, predictions, average='micro')
