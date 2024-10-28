# Multi-Step-Backmapping
Code for multi-step backmapping framework. 

A modified CGVAE architecture is used for CG-CG backmapping, with saved model weights for each coarse-graining experiment and protein found in the CG-CG/ckpts direcrory. 

Three interchangeable models are available for RBCG-AA backmapping and weights are found in the ckpts directory of the corresponding model folders (GenZProt, cg2all, DiAMoNDBack) located in the RBCG-AA folder.

Data for the two example proteins selected, eIF4E, and SARS-CoV-2 PLpro should be downloaded from the following links and placed into the data directory of this repository. The corresponding CG mappings (for use by the CG-CG models) for each experiment performed are already located in the data directory and if desired, can be converted to a coarse-grained trajectory for visualization or for training CG-CG models on additional steps using the scripts in the data/preprocessing directory.

## References
This project "stands on the shoulders of giants" and was made possible due to the following previous models and studies:

Lim Heo & Michael Feig, "One particle per residue is sufficient to describe all-atom protein structures", _bioRxiv_ (**2023**). [Link](https://www.biorxiv.org/content/10.1101/2023.05.22.541652v1)

Michael S. Jones, Kirill Shmilovich, Andrew L. Ferguson, "DiAMoNDBack: Diffusion-denoising Autoregressive Model for Non-Deterministic Backmapping of Cα Protein Traces", _Journal of Chemical Theory and Computation_ (**2023**). [Link](https://arxiv.org/abs/2307.12451)

Soojung Yang & Rafael Gómez-Bombarelli, "Chemically Transferable Generative Backmapping of Coarse-Grained Proteins", _Proceedings of Machine Learning Research_ (**2023**). [Link](https://proceedings.mlr.press/v202/yang23e/yang23e.pdf)

Wujie Wang, Minkai Xu, Chen Cai, Benjamin Kurt Miller, Tess Smidt, Yusu Wang, Jian Tang, Rafael Gómez-Bombarelli, "Generative Coarse-Graining of Molecular Conformations", _International Conference on Machine Learning_ (**2022**). [Link](https://arxiv.org/abs/2201.12176)
