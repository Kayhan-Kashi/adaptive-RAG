---
tags:
- transformers
- xlm-roberta
library_name: transformers
license: cc-by-nc-4.0
language:
  - multilingual
  - af
  - am
  - ar
  - as
  - az
  - be
  - bg
  - bn
  - br
  - bs
  - ca
  - cs
  - cy
  - da
  - de
  - el
  - en
  - eo
  - es
  - et
  - eu
  - fa
  - fi
  - fr
  - fy
  - ga
  - gd
  - gl
  - gu
  - ha
  - he
  - hi
  - hr
  - hu
  - hy
  - id
  - is
  - it
  - ja
  - jv
  - ka
  - kk
  - km
  - kn
  - ko
  - ku
  - ky
  - la
  - lo
  - lt
  - lv
  - mg
  - mk
  - ml
  - mn
  - mr
  - ms
  - my
  - ne
  - nl
  - 'no'
  - om
  - or
  - pa
  - pl
  - ps
  - pt
  - ro
  - ru
  - sa
  - sd
  - si
  - sk
  - sl
  - so
  - sq
  - sr
  - su
  - sv
  - sw
  - ta
  - te
  - th
  - tl
  - tr
  - ug
  - uk
  - ur
  - uz
  - vi
  - xh
  - yi
  - zh
---
Core implementation of Jina XLM-RoBERTa

This implementation is adapted from [XLM-Roberta](https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta). In contrast to the original implementation, this model uses Rotary positional encodings and supports flash-attention 2.

### Models that use this implementation

- [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2)

### Converting weights

Weights from an [original XLMRoberta model](https://huggingface.co/FacebookAI/xlm-roberta-large) can be converted using the `convert_roberta_weights_to_flash.py` script in the model repository.
