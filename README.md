# Imitation Learning for Sentence Generation with Dilated Convolutions Using Adversarial Training

The source code for the paper "Imitation Learning for Sentence Generation with Dilated Convolutions Using Adversarial Training".

<br>

## Prerequisites

- [python >= 3.5.0](https://www.python.org/)
- [tensorflow >= 1.8.0](https://www.tensorflow.org/)

<br>

## Execution

**1.** For data preprocessing :

```
python preprocess.py
```
The preprocessing must be done before training.
The default path of the dataset is **"./dataset/kit.txt"** in **preprocessing.py**.
If you want to use another dataset, please change the **src_path** variable in **preprocessing.py**.
For the format of the dataset, please refer to the file in the **"dataset"** folder.

<br>

**2.** For pre-training word2vec **(Optional)** :

```
python train_word2vec.py
```
The trained parameter will be saved in the **"save_embed"** folder.

<br>

**3.** For pre-training the model with MLE **(Optional)** :

```
python train_mle.py
```
The trained parameter will be saved in the **"save"** folder.

<br>

**4.** For training the model with policy gradient :

```
python train_pg.py
```
The trained parameter will be saved in the **"save"** folder. The code implements the Actor-Critic algorithm for policy gradient training.

<br>

**5.** For testing :

```
python test.py
```
The output sentences with different temperature values will be saved in the **"result"** folder.

<br>

## Experimental Results

- [KIT Motion-Language Dataset](https://motion-annotation.humanoids.kit.edu/dataset/)

Model|Sentences
-|-
Real Sentences|A standing person waves with both hands .<br>A human performs a 90<sup>o</sup> , a 180<sup>o</sup> and a 360<sup>o</sup> counter clockwise jump .
&lambda;=1.00|Somebody walking forward while afterwards three goes forward , first up and ...<br>A passenger cat hydrant is next to a toy establishment the mountains in a signs baby .
&lambda;=0.10|A human walks in a 90<sup>o</sup> curve .<br>The human goes four half steps .
&lambda;=0.01|A human walks a jump to the left .<br>A person jumps to the right and stairs .
&lambda;=0.00|Subject jumps backwards .<br>A person goes left .

<br>

- [MS COCO Dataset](http://cocodataset.org/#home)

Model|Sentences
-|-
Real Sentences|A blue boat themed bathroom with a life preserver on the wall .<br>Two cars parked on the sidewalk on the street .
&lambda;=1.00|A young image in a ground behind many sink in the . area . a field and people to ...<br>A person is walking forward a quarter circle to the left .
&lambda;=0.10|Boat reaches over towards the brick wall .<br>A couple of white people stand along a white bus .
&lambda;=0.01|A blue plate of bags of elephants feeds out of mirror .<br>A motorcycle older walk with a tie in the train .
&lambda;=0.00|A man in a pair of grass setting .<br>A brick table with a plane and shower .
