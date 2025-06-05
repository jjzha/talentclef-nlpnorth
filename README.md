# talentclef-nlpnorth

This repository contains the code to reproduce the results of the NLPnorth team at the TalenCLEF shared task (https://talentclef.github.io/talentclef/).

The repository consists of a single folder that contains the scripts we used to obtain our predictions. The steps that we used (i.e. exact commands) can be found in `scripts/runall.sh`. We do not recommend to run this script, as it will take very long. Instead, use the commands there as a guide for reproducing certain results, and parallelize some of the steps. The scripts have a number as a prefix, which indicate their function:

- 0: preparation of data and setup
- 1: running the classification models
- 2: running the contrastive models
- 3: running the prompt-based models
- 4: evaluate the models
- 5: obtain the results overviews



