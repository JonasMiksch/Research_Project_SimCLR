# Research_Project_SimCLR


Make following changes in the cell beneath the imports:

- Disable mounting of google drive account and unzipping of images

- change working directory, csv paths and image paths accordingly 
      -> for classifier file do the same, also change the name of the pretrained model

- create "checkpoints" folder in your working directory
      -> add model.ckpt from https://drive.google.com/file/d/17h-Hd08n-f_5D8cDV08dpB_-W1cs5jbt/view to the checkpoints folder
      
-set trainingWiki = True and others to false if not already set


in last cell in train_simclr change max_epochs accordingly 
When just loading already trained model set resume = False
