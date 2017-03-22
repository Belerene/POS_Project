Description for POS Tagger

It is necessary to download training corpus from http://eng.slovenscina.eu/tehnologije/ucni-korpus and change the path to the corpus in test_spark_mlp() and test_spark_bayes() functions.
To process the corpus and save the result, uncomment the rows with loadData() and saveData() in test_spark_mlp() and test_spark_bayes() functions. If the processed corpus is already saved comment the loadData() and saveData() out and leave the loadDataFromFile() uncommented.
You can choose to use POS tagger implemented with neural network (NN) or Naive Bayesian Classifier (NBC) by commenting/uncommenting test_spark_mlp() for NN implementation or test_spark_bayes() for NBC implementation at the end of the file.
