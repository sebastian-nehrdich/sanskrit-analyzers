# ByT5-Sanskrit Analyzer 

This repository holds code for training and inference based on the ByT5-Sanskrit model as described in this paper: XXX  

## Data Source

The training data for this model has been taken from the Digital Corpus of Sanskrit (DCS):

[http://www.sanskrit-linguistics.org/dcs/](http://www.sanskrit-linguistics.org/dcs/)

The pretraining data was taken from the [Sangraha dataset](https://huggingface.co/datasets/ai4bharat/sangraha) by AI4Bharat.

## Training

The training code is described in training/.   

## Inference

- Supports multiple processing modes:
  - Segmentation
  - Segmentation with morphosyntactic analysis
  - Lemmatization
  - Lemmatization with morphosyntactic analysis
  - Segmentation, lemmatization, and morphosyntactic analysis combined

## Requirements for Inference

- Python 3.6+
- PyTorch
- Transformers
- tqdm
- pandas

You can install these requirements with pip: `pip install torch transformers tqdm pandas`.  

### Inference Arguments (run_inf.py)

- `--input-file`: Path to the input file containing Sanskrit text (required)
- `--mode`: Processing mode (required)
  - Choices: 'lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'
- `--output-file`: Path to the output file (required)
- `--output-mode`: Output format (optional, default: 'txt')
  - Choices: 'txt' (plain text), 'tsv' (tab-separated values)
- `--batch-size`: Batch size for processing (optional, default: 20)

### Inference Examples

1. Segmentation:

`python run_inf.py --mode segmentation --input-file examples/toy-skt.txt  --output-file examples/toy-skt-segmented.txt`

2. Lemmatization with morphosyntactic analysis in TSV format:

`python run_inf.py --mode lemma-morphosyntax --input-file examples/toy-skt.txt  --output-file examples/toy-skt-analyzed.tsv --output-mode tsv`

## Inference Output

- For 'txt' output mode: Each processed sentence is written on a new line in the output file.
- For 'tsv' output mode: The output is a tab-separated file with columns for segment number, original text, and analyzed text. 
