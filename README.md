Source code for paper : A perspective for Neural Capacity Estimation: Viability and Reliability

How to use:

1- Run the Estimators.py to set different MI estimatition methods.

2-Run mi_critics.py where different MLPs are initialized for mi estimation (which is called ''NMIE'')

3-Run NITs.py.

4-Run the train_all to joint train the NIT_NMIE networks for different channels and parameters as follow:

''typeinp'': refers to the channel type to be used which can be either of : ''conts_awgn'', ''discrt_poisson'',''mac''. 

''seed_size'' refers to the number of mass-points for initial distribution to feed the NITs in the discrete-input channel scenarios.

all other parameters for NITs and trainer function are clear from the names, and the context.

5- To simulate the peak-power contraint/optical AWGN, simply use ''conts_awgn'', and apply the proper condition on the parameters ''peak'' and ''positive''.

6-We have put a separate file for examining the performance of conditional mutual information estimator for the joint NIT_NMIE optimization in ccmi_capestimator.py


