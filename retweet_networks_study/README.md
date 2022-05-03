# Retweet Networks Study 

## Setup

Create Environment
```
pipenv install
```

Activate environment
```
pipenv shell
```

## Directories

### Simulations

The simulations directory contains all simulations performed on the twitter data. There were three main simulations performed (all found in separate .py files for liberals and conservatives):  The mean absolute difference scripts measures how similar chosen peers are to each ego in the dataset compared to a random baseline to establish that both groups exhibit homophily. The acrophily simulation python files simulates both homophily and acrophily to compare the probability of retweeting a more extreme peer to users' empirical choices. The probability difference runs the homophily simulation, finds the probability of retweeting a more extreme peer in both the homophily and empirical conditions, and finds the difference between the probabilities by subtracting the probability in the homophily simulation from the actual probability in the empirical condition. These probability differences were then standardized to create an acrophily coefficient in a separate script (see Data Processing below). All three simulations result in generated data sets or pickle files that were saved to the data folder.

A Jupyter Notebook ("analyze_and_plot") prototyping the acrophily simulation can be found in this directory as well.

### Data processing

This directory contains the script used to merge the probability difference coefficient datasets created in the probability difference simulation python files, as the script was run on 10% of users at a team for both the conservative and liberal groups. It additionally standardizes the probability coefficients by dividing the probability differences by the standard deviation of probability differences overall across both conservatives and liberals.

### Analysis

Scripts containing visualization or evaluation of the datasets can be found here.
