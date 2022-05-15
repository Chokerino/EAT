# EAT

Reproduction is straight forward and implemented by the authors, so I have not added anything in that front. I wanted to see how much improvement has batch hard soft margin loss achieved over using the classic triplet loss. Using the traditional method has one clear disadvantage - we cannot utilize the CATH heirarchies for the data triplets. []

# Train your own network

The following steps will allow you to replicate the training of ProtTucker:
First, clone this repository and install dependencies:

```sh
git clone https://github.com/Rostlab/EAT.git
```

Next, download pre-computed embeddings used in the paper (ProtT5, ProtBERT, ESM-1b, ProSE: 5.5GB in total) to data/ProtTucker and unzip them. Also, download CATH annotations used for training.

```sh
wget -P data/ProtTucker/ https://rostlab.org/~deepppi/prottucker_training_embeddings.tar.gz
tar -xvf data/ProtTucker/prottucker_training_embeddings.tar.gz -C data/ProtTucker/ --strip-components 1
wget -P data/ProtTucker https://rostlab.org/~deepppi/cath-domain-list.txt
```

Finally, start training by running the training script:

```sh
python train_prottucker.py
```

By default, this will train ProtTucker as reported in the paper using embeddings from ProtT5.
In order to change the input embeddings, you can either replace the file name for 'embedding_p' OR compute your own embeddings (supported input format: H5).

# Reference (Pre-print)

```
@article {Heinzinger2021.11.14.468528,
	author = {Heinzinger, Michael and Littmann, Maria and Sillitoe, Ian and Bordin, Nicola and Orengo, Christine and Rost, Burkhard},
	title = {Contrastive learning on protein embeddings enlightens midnight zone at lightning speed},
	year = {2021},
	doi = {10.1101/2021.11.14.468528},
	URL = {https://www.biorxiv.org/content/early/2021/11/15/2021.11.14.468528},
	journal = {bioRxiv}
}
```
