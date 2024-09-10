**Spam Detection Filter of Quora questions**

This project aims to classify text data as spam or not spam using a Long Short-Term Memory (LSTM) deep learning network with pre-trained GloVe embeddings. The dataset used is a CSV file containing questions and their corresponding labels. The model is trained and evaluated on this dataset, with performance metrics reported.

**Data Loading and Preprocessing**

1. **Loading the Data**: The training data is loaded from a CSV file. The dataset contains three columns: `qid`, `question_text`, and `target`.

2. **Reading Word Embeddings**: Pre-trained GloVe embeddings are read from a text file and stored in a dictionary. The embedding vectors are 100-dimensional.

3. **Train-Test Split**: The dataset is split into training and testing sets with a ratio of 80:20.

4. **Tokenization and Sequence Padding**: A `Tokenizer` object is created to tokenize the text data. The text sequences are converted into sequences of integers and padded to a maximum length of 251.

**Model Architecture**

1. **Embedding Layer**: An embedding layer is initialized with the pre-trained GloVe embeddings. The embedding matrix is filled with the pre-trained word vectors.

2. **LSTM Layer**: An LSTM layer with 50 units is added to the model. The `unroll=True` parameter is used to disable cuDNN for better performance.

3. **Dense and Dropout Layers**: A dense layer with 10 units and ReLU activation is added. A dropout layer with a dropout rate of 0.2 is added to prevent overfitting. The final layer is a dense layer with a sigmoid activation function for binary classification.

**Model Training**
 
 The model is compiled with the Adam optimizer and binary cross-entropy loss function. Then, it is trained for 20 epochs with a batch size of 50.
 
**Training and Validation Accuracy**

The model achieved a training accuracy of approximately 95% after 20 epochs. The validation accuracy was approximately 85% after 20 epochs.

**Performance Metrics**

- **Confusion Matrix**: The confusion matrix is used to evaluate the performance of the model and shows the number of true positives, true negatives, false positives, and false negatives. It is then plotted to a heat map for data visualization.

- **Classification Report**: The classification report provides precision, recall, F1-score, and support for each class. The model achieved a precision of 0.85, recall of 0.85, and F1-score of 0.85 for the positive class.

- **ROC AUC Score**: The ROC AUC score is used to evaluate the performance of the model. The model achieved a ROC AUC score of 0.90.
