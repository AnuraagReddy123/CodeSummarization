1. Clone the repo
```
git clone https://github.com/AnuraagReddy123/CodeSummarization.git
```
2. Install the requirements in a virtual environment.
```
cd CodeSummarization/
pip install -r requirements.txt
```

3. Setup tree-sitter-codeviews
```
cd tree-sitter-codeviews
bash setup.sh
```

4. Download the dataset in to Data/dataset.json
```
pip install gdown
mkcd Data
gdown 19PARrtkQ2GfFPodEkkafKogRQAJvQofL -O dataset_all.json
```

5. Run the steps in Data_Preprocessing/README.md
6. Train the model