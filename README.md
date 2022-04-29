# Analysis of Political Tie Selection Data

This repo contains all processing, visualization, analysis and evaluation of political tie selection data gathered from both Twitter and from an MTurk study from "Strategic Identity Signaling" (Van der Does, Tamara et al., 2022), as well processing, simulations, visualizations, analysis and evaluation of twitter data from 147434 conservative users and 274437 liberal users to investigate whether users prefer to select more extreme peers within their own group.

## Directories

### Simulations

The simulations directory contains all simulations performed on the twitter data. There were three main simulations performed (all found in separate .py files for liberals and conservatives):  The mean absolute difference scripts measures how similar chosen peers are to each ego in the dataset compared to a random baseline to establish that both groups exhibit homophily. The acrophily simulation python files simulates both homophily and acrophily to compare the probability of retweeting a more extreme peer to users' empirical choices. The probability difference runs the homophily simulation, finds the probability of retweeting a more extreme peer in both the homophily and empirical conditions, and finds the difference between the probabilities by subtracting the probability in the homophily simulation from the actual probability in the empirical condition. These probability differences will then be standardized to create an acrophily coefficient in a separate script (see Data Processing below).

A Jupyter Notebook ("analyze_and_plot") prototyping the acrophily simulation can be found in this directory as well.

### Data Processing

Due to computational constraints, the probability difference simulation in particular had to be run on ten percent of users in the twitter dataset at a time for both conservatives and liberals. The script that merges these probabilities and standardizes them to create the finalized acrophily coefficient can be found in this directory. Scripts to convert simulation data from the two other simulations, as well as the "Strategic Identity Signaling" paper data, to long format can also be found here.

### Analysis

The analysis directory contains all analysis and evaluation of the data from "Strategic Identity Signaling" as well as analysis and evaluation of the processed data.

### Plots

Scripts to create visualizations based on processed data can be found here.
