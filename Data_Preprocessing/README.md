

1. Install the dependencies:
```
npm i
```

2. Augument the dataset:
```
node main.js
```

* This will read the ```Data/dataset.json``` file and augument the dataset witl **issue titles** and **ASTs** of the modified files. And, finally saves the augumented dataset in ```Data/dataset_aug.json``` file.


3. Preprocess the dataset:
```
python preprocess.py
```
* Preprocesses all the **text** in the dataset using NLTK library.
* Computes the **vocabulary** and saves it in ```vocab.txt```.
* Replaces all the words in the text (including label in ASTs) with arrays of **integers**.
* Removes unnecessary fields in the data points like "id", "cms".
