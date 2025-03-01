# Protein-Stability-Prediction-ML-Challenge
As part of a private kaggle challenge for a grad level machine learning in biosciences class. This model ranked #1 out of 87 participants on the full test data. 
The task in this exercise is a regression problem where we need to develop an ML model to predict a protein's thermostability (a real-valued score) using its amino acid sequence as input.

##  Pre-computed embeddings
I have the pre computed embeddings for the train and test data available at [Google Drive Link ](https://www.example.com](https://drive.google.com/drive/folders/12TDMgEKjjCJd3WSoMhRNWnfcEQBoGao_?usp=sharing))

# ESM2 Model
This project leverages the ESM2 Model (esm2_t6_8M_UR50D) by Meta Research. 
<https://github.com/facebookresearch/esm>

## Background: 
This exercise focuses on predicting the thermostability of proteins, a critical trait that enhances various application-specific functions in the realm of computational biology. With implications ranging from improving reaction rates in enzymatic processes to serving as prime candidates for directed evolution, accurately predicting protein thermostability is a challenge that combines complex biological understanding with computational prowess. 

The dataset for this competition is curated from an extensive screening landscape, utilizing a mass spectrometry-based assay to measure protein melting curves. This dataset encompasses both global and local variations in protein sequences, offering a comprehensive view of how sequence variations influence thermostability.

## Data Analysis and Data Cleaning
<img width="356" alt="image" src="https://github.com/user-attachments/assets/f3c24ee6-f812-4337-83a6-8c198f9eec2d" />

Before beginning the model development process, I wanted to understand what the distribution of data looked like. The above graph clearly shows that the majority of sequences are under 1500 in length, with very few sequences being greater than that. This means that it could be beneficial to use just portions of the sequence to save on computational costs. Thus, I have used the sequences up to 1500 in length. I padded the sequence if the sequences were shorter and cut the sequence at the end if they were greater than 1500. This will ensure that all my sequences are of the same length. 

## Sequence Encoding and Feature Engineering
I experimented with several types of data encodings, In the beginning, I used one-hot encoding as provided in the baseline code, but soon switched to using Label Encoding by sklearn. Label encoding has often been used to embed textual data, so I thought to consider each unique amino acid character as a word.

I then decided to read literature on thermostability prediction using protein sequences. Several papers have used feature engineering to extract different features from sequences and used a combination of those to train their models. I decided to use one of those -the dipeptide frequency and use that as an input to my MLP model. This process resulted in an increase in my Kaggle score to 0.59.

I read some more literature and came across a protein language model called ESM-2 . ESM-2 stands for Evolutionary Scale Model2 is a large language model for protein sequences. It is created by Meta AI’s Fundamental AI Research Team. It has been trained on the Uniref50 database. They have made their pre-trained models available on Huggingface from where I downloaded and used their model. ESM-2 has been known to outperform all tested single-sequence protein language models across a range of structure prediction tasks. I utilized this pre-trained model to generate embeddings from the sequences. 

## 	ML model: Architecture and Rationale
During this challenge, I experimented with several different types of model structures. 
My final model which accepted the output of the ESM2 model, had 1 input layer, 2 hidden layers, and 1 output layer.  I used multiple fully connected layers.
For the activation function, I tested 2 nonlinear functions - ReLu and Tanh, I ended up using Tanh in my final model as it gave me the best scores consistently during hyperparameter testing. I believe that Tanh was the better choice in this scenario because my input features were in a range of -1 to 1. 

For the loss function, considering the nature of this regression task I used mean squared error. Since MSE will calculate the average of the difference between the predicted and observed value, in my opinion, it should correlate very well with the Spearman correlation (with what the model will be tested).

The training of this model involves forward and backward passes, after each layer, I have applied the activation function and dropout regularization except for the last layer which will be fed as output. I calculated validation MSE using 20% of the dataset, to measure the improvement of my model over successive experiments. 

## Model Training
While training my model, I did an 80-20 split. I used 80% of the data to train the model and then used the other 20% of the data as a validation set to understand if my model was underfitting or overfitting. I used Adam as my optimizer and used MSE as a loss function to calculate the gradient descent. I used a variable number of epochs. While doing an initial analysis to see the model’s performance, I trained the model to 1000 epochs. Since I also printed my validation and training loss for each epoch, I could see at what point my model starts to overfit (when train loss is decreasing but validation loss is stable or increasing). On subsequent runs, I used early stopping techniques such as patience of 10 epochs (stop if validation loss does not form a new minimum in 10 iterations) or if I wanted to train up to a certain validation loss (eg break if validation loss is less than 10.7)

Since most of the targets lie in a narrow range of 45-55, I also tried to scale the target values using StandardScaler, run the model, and then convert back the scaled predictions to the original scale. But for some reason, this approach gave me subpar performance when compared to using the target values without any scaling. 
<img width="291" alt="image" src="https://github.com/user-attachments/assets/b01c9e3d-f277-4726-8905-b8dbeb3665d1" />

## Hyperparameter selection
<img width="468" alt="image" src="https://github.com/user-attachments/assets/56a090a6-e9a7-4d0d-8ce1-b6cbf93a3019" />
<img width="328" alt="image" src="https://github.com/user-attachments/assets/f4bc86be-63e5-42c2-b3ca-84226a121fa1" />

For the hyperparameter selection, I used a tool called Optuna, I modified my code in such a way that I was able to run 5000 trials using several hyperparameters. 

I used the following as hyperparameters - 
learning rate, number of epochs, type of optimizer, type of activation function, number of layers in my MLP, batch size, and size of layers. 

I got the following as the best combination of the hyperparameters - 
Learning rate – 0.000025.
Number of epochs -340
Optimizer - Adam
Activation function - Tanh
Number of layers - 2 
Batch size - 64
Size of layers – 2560,1280,100,1





