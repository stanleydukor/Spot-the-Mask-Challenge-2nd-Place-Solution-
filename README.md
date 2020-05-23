## This explains the procedures to reproduce the 2nd place score on the Leaderboard

### Description
Face masks have become a common public sight in the last few months. The Centers for Disease Control (CDC) recently advised the use of simple cloth face coverings to slow the spread of the virus and help people who may have the virus and do not know it from transmitting it to others. Wearing masks is broadly recognised as critical to reducing community transmission and limiting touching of the face.

In a time of concerns about slowing the transmission of COVID-19, increased surveillance combined with AI solutions can improve monitoring and reduce the human effort needed to limit the spread of this disease. The objective of this challenge is to create an image classification machine learning model to accurately predict the likelihood that an image contains a person wearing a face mask, or not. The total dataset contains 1,800+ images of people either wearing masks or not.

Competition Link - https://zindi.africa/hackathons/spot-the-mask-challenge

This process follows the heading for each cell in the notebook.

### 1. If using Colab
If using Google Colab, run each cell

### 2. Import Libraries
Imports all necessary libraries used also well as setting seeds and specifying directories

### 3. Create Folds
Run this cell to create a new dataframe for K fold crossvalidation. (Necessary)

### 4. Set Environment Variables
Create cuda variable for moving model and data to GPU and setting batch sizes for train and test

### 5. Load Train Data From DataFrame
from_df class for loading and passing data into dataloader.

### 6. Model Building
Defined classes for each pretrained models to be used and also a model dispatcher dictionary

### 7. Training
Defined all training parameters.

### 8. Train on Full Dataset
Train using Se_ResNext101_32x4d for 50 epochs and save model, train using SeNet154 for 50 epochs, save model and train for extra 50 epoch and save second model.

### 9. Load Model
Load the three models seperately.

### 10. Testing
Test and generate 3 submission files using the three models.

### 11. Open ensemble.ipynb and perform model averaging
Load the three submission files from the three models and find the average, and submit.
NOTE: This will return a score slightly higher than the one on the leaderboard. Averaging se_resnext101_32x4d and senet154_100 gives the score on the leaderboard. 