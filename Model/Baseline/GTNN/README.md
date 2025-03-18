# GTM
This is the tensorflow implementation of the paper "Graph Topic Neural Network for Document Representation"

## Implementation Environment
- Python == 3.6
- Tensorflow == 1.9.0
- Numpy == 1.17.4

## Run
`python main.py`

### Parameter Setting
- -lr: learning rate, default = 0.004
- -ne: number of epoch, default = 6000
- -dn: dataset name
- -nt: number of topics, default = 64
- -tr: training ratio, this program will automatically split 10% among training set for validation, default = 0.8
- -ms: minibatch size, default = 128

## Data
Some codes and four subset datasets of cora are reference from the paper https://github.com/cezhang01/Adjacent-Encoder

## Output
The document embeddings are output to the `./results` file. Each row represents one document embedding, and each column represents one dimension of the embedding, or one topic.
