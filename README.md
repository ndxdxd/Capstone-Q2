# Capstone Quarter 2 Project: Adversarial Attacks on Invisible Watermark

By Anushka Purohit and Andy Truong

This project innovates a new watermarking scheme, creating an invisible watermark inspired through adversarial attacks. Specifically, Projected Gradient Descent where we add small pertubations onto an image to misclassify it.

This project explores the question: Can we create a watermark that is robust to regeneration without greatly sacrificing the quality of an image.


## Running the project

First git clone the repository on your local machine, and make sure that you have a gpu that can run the notebooks. 

To install the dependencies, run the following command from the root directory of the project: pip install -r requirements.txt

##  Building the project stages using run.py.
Due to the nature of the size of the data, we can not publish it onto Github, but we are able to give the code to download it.
To get the data, from the project root dir, run python run.py. This fetches the images from kaggle, creates saves the data in the imagenet1k-val folder.

## Recreating our Results
There are 5 different notebooks in the notebook folder which we used to generate our results. Simply run through those notebooks to see the watermarking process as well as the visualizations we created.  To run our project comprehensively, run the black box and white box attack notebook files first. Make sure you have the requirements installed first. 

## License and Attribution

The PGD attackis based on [adversarial-learning-robustness](https://github.com/dipanjanS/adversarial-learning-robustness?tab=readme-ov-file) and includes code licensed under the Apache License 2.0.

We also again used our mentor's regeneration attack. 

WatermarkAttacker: By Xuandong Zhao @article{zhao2023invisible, title={Invisible Image Watermarks Are Provably Removable Using Generative AI}, author={Zhao, Xuandong and Zhang, Kexun and Su, Zihao and Vasan, Saastha and Grishchenko, Ilya and Kruegel, Christopher and Vigna, Giovanni and Wang, Yu-Xiang and Li, Lei}, journal={arXiv preprint arXiv:2306.01953}, year={2023} }