# CofCED_running codes

## 1. Installing requirement packages
```
conda create -n fact22 python=3.8
source activate fact22
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install transformers pandas==1.1.2 tqdm==4.50.0 nltk==3.5 rouge-score==0.0.4 sklearn
pip install sentence_transformers   # for evaluation
pip install torch>=1.8
```
Tips: - Adding a `logs` dir to the path of `datasets`. 

## 2. Follow the guide to download the datasets and put them in the correct location. 

## 3. Run the code
It is recommended to run on linux servers with the following script: 
`python train_exp_fc5_xxx.py`

## 3. Please cite this paper as follows （BibTeX）: 
```
@inproceedings{yang2022cofced,
  title={A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection},
  author={Yang, Zhiwei and Ma, Jing and Chen, Hechang and Lin, Hongzhan and Luo, Ziyang and Chang Yi},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics (COLING)},
  pages={2608--2621},
  month={oct},
  year={2022},
  url={https://aclanthology.org/2022.coling-1.230},
}
```

PDF: https://aclanthology.org/2022.coling-1.230.pdf
