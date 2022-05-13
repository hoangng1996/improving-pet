# Improving-PET
This is the final project of Group 10 for the course Natural Language Understanding and
Computational Semantics Spring 2022. \
In this project we aim to improve upon Patter-Exploiting Training (PET) by introducing
data augmentation techniques as well as  multi-tasks tranining

### Structures

There are 3 main components: PET, AdaPET and Data Augmentation, each with their individual
README file from the original authors of the paper, viewers can follow the instruction
to get the example result


### Pipeline

![Alt text](pipeline.png?raw=true "Pipeline")

We use EDA to generate new training examples from original dataset before feeding in
to Aug-PET model

### Result
|  Examples| Method | AG's | Yahoo Answer  |
| :---: | :---: | :---: | :--- |
| τ = 10| PET |  86.7% | 63.6% |
| τ = 10| Aug-PET |  88.5%  | 64.4% |
| τ = 50| PET |  86.6% | 65.3% |
| τ = 50| Aug-PET |  86.6% | 65.9% |
| τ = 100| PET |  88.1%  | 68.3%|
| τ = 100| Aug-PET |  89.0%  | 69.1% |




 







We present the results in ascending order of size of labelled data