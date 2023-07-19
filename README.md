# Deep Cascade-Learning for Immunofixation Electrophoresis (IFE) Analysis

Immunofixation Electrophoresis (IFE) analysis has been an indispensable prerequisite for the diagnosis of M-protein, which is an important criterion to recognize diversified plasma cell diseases. Based on the feature structure and diagnostic basis of IFE images, we develop a new deep cascade-learning model for identifying M-protein existence and its isotype. Our contributions can be concluded as follows:

(1) We develop a cascade-learning framework, which leverages two cascade models to deal with the two classification subtasks according to their own characteristics. By optimizing two subtasks respectively, the performance of cascade framework can reach the best;

(2) We employ recurrent attention model (RAM) to mimic the human inspection procedure of IFE data, i.e., only paying attention to the horizontal locations with potential dense bands, instead of the whole image. SP lane is also considered in RAM to guide its location network, which helps better and faster search of dense bands;

(3) We further propose two modified glimpse mechanisms in RAM framework, which can adaptively deploy glimpse sensors with limited bandwidth in the most informative regions to avoid redundancy and save computational power.

This repository contains our demo codes of Cas-TD-RAM (Cas-OD-RAM) and simulated dataset of IFE images:

(1) Code: Our demo codes are merely simple implementations of three modules in Cas-OD-RAM and Cas-TD-RAM, which includes negative-positive classifier, positive-isotype classifier, and cascade test. We will release our overall code once our manuscript is accepted, to support further exploration of the IFE problems.

(2) Dateset: Due to data privacy and security concerns, we kindly regret that we cannot release our original dataset. As an alternative, we construct a simulated dataset through generative models whose size is comparable to our original dataset.


# Quick Tour
This is a demo implementation of the deep cascade-learning method proposed in our TMI manuscript [1].

Environment: Python 3.6.10; tensorflow-gpu 1.3.0.

(1) Code "trainmodel_DCL" describes our negative-positive classifier, which references the PyTorch code for deep collocative learning (available at https://github.com/lookwei/collocative-learning-4-IFE) [2]. To run this code, we can obtain the negative/positive probabilities of each sample.

(2) Code "ODRAM_train_coa" ("TDRAM_train_coa") describes our positive classifier. To run this code, we can obtain the eight-isotype probabilities of each sample.

(3) Code "cascade_demo" describes the result combination of DCL and OD-RAM (TD-RAM). To run this code, we can obtain the overall nine-class probabilities of each sample.

# Dataset
Due to the privacy issue, we cannot distribute the original IFE dataset used in the paper "Deep Cascade-Learning Model via Recurrent Attention for Immunofixation Electrophoresis Image Analysis". However, we create a simulated dataset that owns similar distributions as the original one. Our clinicians have gone through the dataset to make sure that it resembles the original one to the maximum extent. The dataset is in the file "IFE_Simulated_Data". We hope it can help initiate your IFE study and verify your methods. The distribution of our dataset is as follows.

| Type | IgAk | IgAL | IgGk | IgGL | IgMk | IgML | k | L | Negative |
| --- | --- | --- | --- |--- | --- | --- | --- | --- | --- |
| Number | 514 | 1086 | 1107 | 1385 | 349 | 156 | 148 | 271 | 3608 |

Because the number of negative samples is so large (over 17000), we will release them in batches. Now, 3608 negative samples can be available. 

# Reference
[1] X. An, P. Li, and C. Zhang, “Deep Cascade-Learning Model via Recurrent Attention for Immunofixation Electrophoresis Image Analysis,” IEEE Transactions on Medical Imaging, under revision, 2023.
[2] X. -Y. Wei et al., “Deep collocative learning for immunofixation electrophoresis image analysis,” IEEE Transactions on Medical Imaging, vol. 40, no. 7, pp. 1898-1910, Jul. 2021.
