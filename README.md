# Research_Project_SimCLR

Pretraining on LME account: Use Pretraining.py
-change working directory, csv paths and image paths accordingly 
-set EITHER 
      trainingSimCLR for the original method,
      trainingArtDl for Style Transfer on artdl
      or
      trainingWiki for Style Transfer on wiki arts
 to True others to False
      
- create "checkpoints" folder in your working directory
      -> add model.ckpt from https://drive.google.com/file/d/17h-Hd08n-f_5D8cDV08dpB_-W1cs5jbt/view to the checkpoints folder
      
      
Finetuning on Classifier_SimCLR

- Disable mounting of google drive account and unzipping of images

- change working directory, csv paths and image paths accordingly 
      -> for classifier file do the same, also change the name of the pretrained model

- rename model to pretraining model
      
