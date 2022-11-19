

1. Augument the dataset with extra artifacts:
```
python augment.py
```

2. Preprocess the dataset:
```
python preprocess.py
```
* Preprocesses all the **text** in the dataset using NLTK library.
* Computes the **vocabulary** and saves it in ```vocab.txt```.
* Replaces all the words in the text (including label in ASTs) with arrays of **integers**.
* Removes unnecessary fields in the data points like "id", "cms".
