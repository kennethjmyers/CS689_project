# Analysis of Robustness in Gradient Boosted Models

An implementation of [VeriGB](https://arxiv.org/abs/1906.10991)

If reading this somewhere else see the [Github repo](https://github.com/kennethjmyers/CS689_project). 

Our main python functions are located in [VeriGB.py](./VeriGB.py). Experiments are located as follows:

- Experiment replication (repeating original experiment) - ExperimentReplication.ipynb
- Input standardization experiment (standardizing inputs and running same tests) - InputStandardization.ipynb
- Results for the above two experiments and the relationship between generalizability and robustness are shown in Results.ipynb
- Experiment for training on adversarial counter-examples and results - CounterExamples.ipynb

Results - results are a collection of pickle files containing data structures holding the results and are located in the following folders:

- Experiment Replication: ./ReplicationResults/
- Input Standardization: ./StandardizationResults/
- Counter Examples: ./CounterExamplesResults/

Dataset: [HSKC Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction)
