# BERT + Masked CRF

A BERT + Masked CRF for Named Entity Recognition (NER). A Masked CRF is added to enforce label transition in a sentence, and allow recognition of out-of-vocabulary (OOV) words by masking loss of `O` labels.

## Requirements

-  `python3`
- `pip3 install -r requirements.txt`

## Run

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=output --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.4`

## Pretrained model download from [here](https://drive.google.com/file/d/1hmj1zC6xipR7KTT04bJpSUPNU1pRuI7h/view?usp=sharing)

## Inference

```python
from bert import Ner

model = Ner("output/")

output = model.predict("Steve went to Paris")

print(output)
# ('Steve', {'tag': 'B-PER', 'confidence': 0.9981840252876282})
# ('went', {'tag': 'O', 'confidence': 0.9998939037322998})
# ('to', {'tag': 'O', 'confidence': 0.999891996383667})
# ('Paris', {'tag': 'B-LOC', 'confidence': 0.9991968274116516})

```

## Original version
https://github.com/kamalkraj/BERT-NER