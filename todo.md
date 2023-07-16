# Big fine-tuning MLP issue

 - Visualize predicted masks of 3rd, 4th images based on the 1st, 2nd images. This should maybe tell you if bad training is because of sample diversity/clustering issues, or just something about IoU for large sample sizes.

 - Run your algo for 50 images with 64 hidden dims. This should be enough, right?

 - Ask people why 1 hidden dim converges better than 8 hidden dims on the same # of input channels.

 - Run a clustering algo (i.e. T-SNE) on avg. embeddings of all your masks for each class. Are they clustered into a few locations? Or is it really all over the place?

 - Note for the future: maybe just use the bitmask and feature map of previous examples, then attend to that in the mask decoder. Would have to train on i.e. SAM-1B, but might seriously work.

 - Try asking the model to predict different gt target similarity maps--for pixels with positive cosine similarity to avg. target embedding, but which aren't in the mask, ask them to be 0.5.

 - Use more layers than 2?

 - Run a big Sweep on hyperparams for this Sim fine-tuning thing.

   - LR
   - \# of hidden dims
   - \# of example images