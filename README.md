## T2I-FineEval: Fine-Grained Compositional Metric for Text-to-Image Evaluation

T2I-FineEval is a novel evaluation metric designed to assess the compositional alignment between textual prompts and generated images in text-to-image models. Unlike traditional metrics, T2I-FineEval decomposes both text and image into fine-grained components, enabling a more precise evaluation of attribute binding and spatial relationships.

## Paper

For an in-depth understanding of the methodology and experiments, refer to the paper:

**T2I-FineEval: Fine-Grained Compositional Metric for Text-to-Image Evaluation**  
*Seyed Mohammad Hadi Hosseini, Amir Mohammad Izadi, Ali Abdollahi, Armin Saghafian, Mahdieh Soleymani Baghshah*  
[arXiv:2503.11481](https://arxiv.org/abs/2503.11481)

## Installation

1. **Clone the Repository**:

```bash
git clone https://github.com/hadi-hosseini/T2I-FineEval.git
cd T2I-FineEval
```

2. **Set Up Environment**:

Make sure you have Conda installed:

```bash
conda env create -f environment.yaml
conda activate comtie
```


3. **Download Required Models**:

Download YOLOv9 weights (see yolov9/ directory for instructions).

Download BLIP-VQA pretrained weights (see src/ directory for details).


## Contact

For questions or collaborations, contact:

Seyed Mohammad Hadi Hosseini

hadi.hosseini17@sharif.edu
