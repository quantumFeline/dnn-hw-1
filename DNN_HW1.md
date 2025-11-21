Dataset exploration and augmentation
---

We will create a class `Dataset` to hold the necessary data. It will handle all the dataset preparation - reading the images and the labels, transforming the data into the suitable format, and augmenting it.

Here, we implement four augmentations - horizontal flip, vertical flip, rotation, and gaussian noise. For each of the augmentations, we choose a random subset of the dataset (50% by default) to apply the augmentation. This means that some images may have multiple augmentations applied, which is intended.

For rotation, as it is unspecified whether we use clockwise rotation, counter-clockwise, or both, we choose the direction on the rotation randomly for each image chosen for this augmentation type.

For Gaussian noise, we will use variance of 10% of the maximal value of the pixel.

After dataset augmentation, we can see that the labels have changed correspondingly:

![dataset before and after augmentation](./dataset_head_before_after.png)

Random sample of augmented images:

![dataset sample after augmentation](augmented.png)

Model architecture
---

We will keep the architecture in the class `Model`.

Training
---