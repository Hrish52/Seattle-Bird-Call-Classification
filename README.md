# Seattle-Bird-Call-Classification
**Project Overiew**
In this project, we aim to classify bird species in Seattle based on their audio calls using a convolutional neural network (CNN) based on a set of audio recordings. The dataset consists of spectrograms which have been derived from audio recordings of birdcalls sounded by 12 different species.

![Birds](https://github.com/Hrish52/Seattle-Bird-Call-Classification/blob/main/Birds.png)

The following two models have been developed:
1. **Binary Classification Model**: Distinguishes between two specific bird species—Northern Flicker and Steller’s Jay.
2. **Multi-class Classification Model**: Identifies calls from 12 bird species found in the Seattle area.

The models were trained on spectrograms generated from bird call audio files using deep learning techniques, with an emphasis on accuracy and efficient classification.

**Dataset**
* **Source**: The dataset was taken from the Xeno-Canto bird sound archive via the Birdcall competition dataset on Kaggle.
* **Data Format**: The dataset contains MP3 recordings of bird calls which were converted into spectrograms for input into the neural network models.
* **Preprocessing**: Audio files were resampled, split, and converted into spectrogram images, which represent the frequencies of bird calls over time. These spectrograms are then normalized to be fed into the CNN models.

**Model Architecture**
**Binary Classification Model**
* **Input**: Spectrograms of bird calls.
* **Layers**:
  * Convolutional layers with ReLU activation.
  * MaxPooling layers to reduce spatial dimensions.
  * Dropout layers to prevent overfitting.
  * Fully connected (Dense) layers to classify the input into one of the two bird species.
* **Output**: Binary classification (Northern Flicker vs. Steller’s Jay).
* **Optimizer**: Adam.
* **Loss Function**: Binary cross-entropy.

**Multi-class Classification Model**
* **Input**: Spectrograms of bird calls.
* **Layers**:
  * Multiple convolutional layers with ReLU activation.
  * MaxPooling layers to reduce spatial dimensions.
  * Dropout layers for overfitting prevention.
  * Fully connected (Dense) layers to classify input into one of 12 bird species.
* **Output**: Multi-class classification (12 species).
* **Optimizer**: Adam.
* **Loss Function**: Categorical cross-entropy.

**Data Preprocessing**
1. **Spectrogram Generation**: Each audio file is converted into spectrograms using librosa, which generates visual representations of sound frequencies over time.
2. **Normalization**: The spectrogram data is normalized to ensure that the CNN models receive inputs within a standard range.
3. **Label Encoding**: The bird species are labeled numerically, with binary labels for the two-species classifier and categorical labels for the 12-species model.
4. **Data Augmentation**: The audio files were split into overlapping windows to increase the sample size and create better generalization for the models.

**Training**
* **Binary Model**: The binary model was trained on 70% of the dataset, with 30% used for validation. The model was trained for 20 epochs using a batch size of 128.
* **Multi-class Model**: Similar to the binary model, this model was trained on 70% of the dataset for 20 epochs, using a categorical cross-entropy loss function. A second model was trained with additional layers for improved performance.
* **Hyperparameter Tuning**: Parameters such as the number of filters, kernel size, learning rate, and dropout rates were fine-tuned to optimize model performance.

**Results**
* **Binary Classification Model**:
  * The model demonstrated strong performance in distinguishing between Northern Flicker and Steller’s Jay, achieving near-perfect classification during evaluation.
* **Multi-class Classification Model**:
  * The model successfully classified calls from 12 species, though certain species, such as the Red-Winged Blackbird and Barn Swallow, were harder to distinguish due to similar call frequencies. Two variations of the multi-class model were trained, one with two convolutional layers and the other with four, resulting in improved accuracy for the latter.

**Predictions**
The multi-class model was further tested on external bird call clips, and the predictions were made using the trained model:
  * Audio clips were processed into spectrograms and input into the model.
  * The predicted bird species and their confidence scores were output for each test clip.

**Performance Metrics**
* **Binary Model**:
  * Precision, Recall, F1-Score
  * Confusion Matrix to show class-wise performance.
* **Multi-class Model**:
  * Accuracy across 12 bird species.
  * Confusion Matrix to show the model's performance on each species.
    
**Challenges**
* **Audio Data Complexity**: Classifying similar-sounding species (e.g., Barn Swallow vs. House Finch) proved challenging due to overlapping call frequencies.
* **Computational Resources**: The models required significant computational resources for training, particularly the multi-class classification model, which took longer due to the larger dataset.

**Future Work**
* **Improving Model Accuracy**: Future work could explore using other machine learning algorithms like Support Vector Machines (SVM) or Random Forests to handle noisy data more efficiently.
* **Real-time Classification**: The models could be extended to classify bird calls in real-time, contributing to more effective biodiversity monitoring.

**References**
1. Dataset:
   * https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m
   * https://xeno-canto.org/ 
2. Tensor Documentation:- https://www.tensorflow.org/guide
3. Librosa Documentation:- https://librosa.org/doc/latest/index.html
4. Audio Classification Project using CNNs:- https://github.com/jeffprosise/Deep-Learning/blob/master/Audio%20Classification%20(CNN).ipynb
