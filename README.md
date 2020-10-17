# Analysis of Robustness in Gradient Boosted Models

An implementation of VeriGB

Look at the notebook I pushed. I commented the algorithm we need to write along with some info on how the predicates should look like. I need to read more into how Z3 works and how the predicates can all be written (functions, conjunctives and disjunctives, etc.).

Also download the data to the data folder and unzip, if it's really important maybe we can put this in a shell script to download this but I'm not interested in doing that right now.

Let's just focus on getting the regression working because it seems easier to code, it has lower data complexity (so faster to solve) and we don't have to worry about parallelization (which we could optimize for classification see Section 6.2 of the paper).

[HSKC Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction)

To do:
- Code out the algorithm the generates predicates of model for solver
- run solver on model
- run two experiments for
  - Correlation between robustness and generalizability
  - Training on adversarial examples
- add requirements.txt file (Z3, sklearn, pandas, numpy, etc)
- shell script that downloads data and unzips, installs requirements, would like to see this work in a clean environment
