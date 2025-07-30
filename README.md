# CoNLL-U UD Parser
This repository contains the code for a Universal Dependencies parser to create probing datasets, based on SentEval.
The code script titled "Connlu_UD_parser" takes the languages treebanks from Universal Dependenciies and converts them from the conllu format to SentEval format for efficient probing. The script consists the code to convert any UD CoNLL-U language dataset to create probing datasets for the tasks mention in Conneau et al. (2018). The tasks include: Sentence length, Word content, Bigram shift, Tree depth, Tense, Subject number, Object number, Semantic odd man out, and Coordination inversion.


---

## 📚 **Features**

This parser supports:
- ✅ Sentence length binning  
- ✅ Dependency tree depth computation  
- ✅ Subject and object number extraction  
- ✅ Tense extraction  
- ✅ Bigram shift generation  
- ✅ Word content extraction  
- ✅ Odd-man-out (SOMO) task generation  
- ✅ Coordination inversion task generation  
- ✅ Automatic data splitting (train, test, validation)  
- ✅ Output writing for SentEval-ready format

---

## **Requirements**

Make sure you have the following Python packages installed:

```bash
pip install numpy conllu tqdm
```
---
To use the parser, you need a mapping of languages and their CoNLL-U file paths.
```python
from Conllu_UD_Parser import Conllu_UD_Parser

file_mapping = {
    "{name of language}": [
        "{path to UD language train data}",
        "{path to UD language dev data}"
    ],
    # add subsequent languages, if necessaary
    }
parser = Conllu_UD_Parser(file_mapping=file_mapping)
output_paths = parser.process()
print(output_paths)
```
