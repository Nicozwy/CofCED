<p align="center">
 <img width=600px src="https://github.com/Nicozwy/CofCED/blob/main/logo.png" align="center" alt="CofCED" />
 <h2 align="center">Wisdom of crowds: CofCED</h2>
 <p align="center"> </p>
</p>

 <p align="center"> :triangular_flag_on_post:  The codes and datasets have been uploaded! </p>

`A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection` is accepted by COLING 2022. 
`CofCED` is an explainable method proposed by this paper. We present the first study on explainable fake news detection directly utilizing the wisdom of crowds (raw reports), alleviating the dependency on fact-checked reports.

:triangular_flag_on_post: If possible, could you please star this project. :star:  :arrow_upper_right:

### Codes 
#### Installing requirement packages
```
conda create -n fact22 python=3.8
source activate fact22
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install transformers pandas==1.1.2 tqdm==4.50.0 nltk==3.5 rouge-score==0.0.4 sklearn
pip install sentence_transformers   # for evaluation
pip install torch>=1.8
```

### Datasets 
We constructed two realistic datasets, i.e., RAWFC and LIAR-RAW, consisting of raw reports for each claim.
- [RAWFC](https://github.com/Nicozwy/CofCED/tree/main/Datasets/RAWFC)
- [LIAR-RAW](https://github.com/Nicozwy/CofCED/tree/main/Datasets/LIAR-RAW)

### Please cite this paper as follows （BibTeX）: 
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


