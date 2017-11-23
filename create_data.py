import pandas as pd
import numpy as np
import math
import pdb
import random

LABELS = {'guess_passwd.': 6, 'nmap.': 17, 'loadmodule.': 2, 'rootkit.': 22, 'warezclient.': 20, 
            'smurf.': 5, 'pod.': 7, 'neptune.': 4, 'normal.': 0, 'spy.': 21, 'ftp_write.': 12, 'phf.': 16, 
            'portsweep.': 9, 'teardrop.': 8, 'buffer_overflow.': 1, 'land.': 11, 'imap.': 14, 'warezmaster.': 19, 
            'perl.': 3, 'multihop.': 18, 'back.': 13, 'ipsweep.': 10, 'satan.': 15}

def dataset_count(N, training_percent, validation_percent):
  training_count = (int)(training_percent * N)
  validation_count = (int)(validation_percent * N)
  test_count = N - training_count - validation_count
  return training_count, validation_count, test_count

def partition_data(indices, training_count, validation_count, test_count):
  indices = np.random.permutation(indices)
  training_samples = random.sample(indices, training_count)
  indices = [index for index in indices if index not in training_samples]
  validation_samples = random.sample(indices, validation_count)
  indices = [index for index in indices if index not in (training_samples + validation_samples)]
  test_samples = random.sample(indices, test_count)
  return training_samples, validation_samples, test_samples

def create_dataset(samples, labels, training_indices, validation_indices, test_indices):
  training_samples, validation_samples, test_samples, training_labels, validation_labels, test_labels = [], [], [], [], [], []
  writer = open('../data/training_set.csv', 'w')
  label_writer = open('../data/training_labels.txt', 'w')
  for index in training_indices:
      writer.write(samples[index] + '\n')
      label_writer.write(str(labels[index]) + '\n')
  writer.close()
  label_writer.close()
  writer = open('../data/validation_set.csv', 'w')
  label_writer = open('../data/validation_labels.txt', 'w')
  for index in validation_indices:
      writer.write(samples[index] + '\n')
      label_writer.write(str(labels[index]) + '\n')
  writer.close()
  label_writer.close()
  writer = open('../data/test_set.csv', 'w')
  label_writer = open('../data/test_labels.txt', 'w')
  for index in test_indices:
      writer.write(samples[index] + '\n')
      label_writer.write(str(labels[index]) + '\n')
  writer.close()
  label_writer.close()

def get_label(label):
  return LABELS[label]

def get_dicts(dataset):
  dict_1, dict_2, dict_3= {}, {}, {}
  count1, count2, count3 = 0, 0, 0
  for sample in dataset:
    value = sample[1]
    if(value not in dict_1):
        count1 = count1 + 1
        dict_1[value] = count1
    value = sample[2]
    if(value not in dict_2):
        count2 = count2 + 1;
        dict_2[value] = count2
    value = sample[3]
    if(value not in dict_3):
        count3 = count3 + 1
        dict_3[value] = count3
  return dict_1, dict_2, dict_3

def main():
  smurf_indices = []
  neptune_indices = []
  normal_indices = []
  other_indices = []
  index = 0
  data = pd.read_csv('../data/kddcup.data_10_percent', sep=',',header=None).values
  for row in data:
    label = row[41]
    if label == "normal.":
      normal_indices.append(index)
    elif label == "smurf.":
      smurf_indices.append(index)
    elif label == "neptune.":
      neptune_indices.append(index)
    else:
      other_indices.append(index)
    index = index + 1

  #normal_indices = random.sample(normal_indices, 100000)
  #smurf_indices = random.sample(smurf_indices, 100000)
  #neptune_indices = random.sample(neptune_indices, 100000)
  samples = normal_indices + smurf_indices + neptune_indices + other_indices

  dataset = []
  for sample in samples:
    dataset.append(data[sample])
  dict_1, dict_2, dict_3 = get_dicts(dataset)

  for sample in dataset:
    sample[1] = dict_1[sample[1]]
    sample[2] = dict_2[sample[2]]
    sample[3] = dict_3[sample[3]]

  N = len(dataset)
  training_count, validation_count, test_count = dataset_count(N, 0.6, 0.2)
  training_indices, validation_indices, test_indices = partition_data(range(N), training_count, validation_count, test_count)

  samples, labels = [], []
  for sample in dataset:
    data = []
    for d in sample[:-1]:
      data.append(str(d))
    samples.append(','.join(data))
    labels.append(get_label(sample[41]))
  create_dataset(samples, labels, training_indices, validation_indices, test_indices)

if __name__ == "__main__":
  main()