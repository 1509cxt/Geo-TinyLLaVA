## Plane Geometry Diagram Formalization via Vision-Language Models

Code and data for the paper "Plane Geometry Diagram Formalization via Vision-Language Models".

Dataset [GDF86K](https://huggingface.co/datasets/1509cxt/GDF86K), Model Checkpoints [Geo-TinyLLaVA](https://huggingface.co/1509cxt/Geo-TinyLLaVA).

![ex1](assets/overview.png)

## Installation and Requirements

1. Clone this repository and navigate to the folder

```bash
git clone https://github.com/Geo-TinyLLaVA/Geo-TinyLLaVA.git
cd Geo-TinyLLaVA 
```

2. With Python 3.10+ and python3-pip, create and activate a virtualenv environment or conda environment, or just use your system Python interpreter 

3. Install all required python dependencies of TinyLLaVA and Inter-GPS

```bash
# enable PEP 660 support
pip install --upgrade pip  

# prerequisites of the training and inference of Geo-TinyLLaVA
cd TinyLLaVA_Factory 
pip install -e .         

# prerequisites of the evaluation of geometry diagram formalization performance and geometry problem solving performance of Inter-GPS
cd ..
pip install -r InterGPS/requirement.txt      
```


## Geometry Diagram Formalization Performance Evaluation

## Geometry Problem Solving Performance Evaluation

## Run Geo-TinyLLaVA 

## Train Geo-TinyLLaVA from Scrach


## Acknowledgement
The project is built on top of [InterGPS](https://github.com/lupantech/InterGPS) and [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory). Thanks for their wonderful works.