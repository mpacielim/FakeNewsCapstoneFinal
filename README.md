## Using Text Classification in Recurrent Neural Networks to Detect Fake News

**Author: Melissa Paciepnik**

### Executive summary

**Project Overview and Goals**

The goal of this project is to create a model that accurately classifies news as Real or Fake based on its linguistic contents.  The motivation behind this is to review the effectiveness of machine learning models at classifying high stakes information such as news.  In a world that is irreversibly entrenched in social media and AI tools, it is crucial to build trust in verified news sources and stem the spread  of misinformation.  In reality we know that this is a multifacted effort that requires media and societal integrity, regulation policies, and general education.  However it is equally important to both find ways to create reliable tools for flagging fake news, as well as educating the public on the limitations of such methods, since there is a high societal cost to classifying both fake and real news incorrectly. 

**Findings**

The best model for correctly classifying news was a recurrent neural network (RNN) model, with an accuracy of 0.872, F1 score of 0.868, recall of 0.913, precision of 0.827 and PR-AUC of 0.954 on unseen test data.  This was trained on a dataset of 750 text samples that were classified as 0 (Real) or 1 (Fake).  The scores mean that model on the 250 new test samples, the model correctly classified 87.2% of samples, with 22 False Positives (Real news incorrectly identified as Fake) and 10 False Negatives (Fake news incorrectly identified as Real).  Note that due to the inherent nature of neural networks, the results in the jupyter notebook change slightly each time it is run, however the RNN model still far outperforms the other models each time.

This model was chosen as the best performing due to having the highest F1 score on test data out of all the models tested.  F1 score was chosen as being the most important in this dataset that was slightly unbalanced (with an approximately 55/45% split between Real and Fake news) as it indicates the best balance of correctly identifying both classes correctly.  This is desirable given that there is a high societal cost to both:
- a) False Positives - incorrectly identifying real news as fake (undermines legitimate news sources and spreads distrust).
- b) False Negatives - incorrectly identifying fake news as real (further spreads misinformation and unnecessary hysteria).
    
5 models were evaluated in total.  1 RNN, and 4 non neural machine learning methods, being Logistic Regression, Naive Bayes, Decision Trees and Support Vector Classification (SVC).

The RNN model was further investigated using the Lime Explainer package in python, which is able to take a sample from the dataset, run it through the model and visually display which words from the sample were most influential in the models output.  This visualizer is a useful tool for providing high level insight to the public on how the data influences the models decisions, which is useful when considering its limitations.

**Next Steps and Recommendations**

It was proven that a neural network was far more effective at analysing large bodies of text than the other four non neural machine learning methods.  In order to further improve this rudimentary RNN model, the model should be trained with pre-trained large language models such as BERT.  It would be hugely beneficial to use this kind of pre-trained neural network that has benefited from vast amounts of data and tuning to more successfully identify patterns in our dataset.   Further tuning can then occur.

Further work could be conducted to make the these models and the visualization of their mechanisms (such as through Lime Explainer) accessible through apps or social media, so that users can both use the models, and visualize what has led the model to that decision.  The more prevalent the models and the accuracy of the visualizations, the better the education effort.

**Final Thoughts**

Ultimately, the negative impact of misinformation on society cannot solely be controlled by algorithms.  It requires societal integrity, widespread education on critical thinking, and understanding biases that are inherent in both real and fake news.  However, it is important to work towards making news sources more transparent, and develop tools that can assist with analysing and cross checking viral information.  The practicality and usefulness of these tools hinges on the quality and lack of bias in the information used to train them, as well as having those that deploy and use such tools understand exactly how they work or atleast their limitations.  Projects such as these are intended to help educate, inform and continue the discussion on the use of AI for fact checking and information distribution.


### Research Question
The goal of this project is to determine the best model for predicting whether or not news articles are from a fake or verified sources based on the linguistic contents of the article.

### Rationale

The importance of my project has two primary parts:

1. It continues the discussion on building trust in society through developing and providing effective tools to be able to critically analyze the validity of the extreme amount of information being shared online. 
    
	- This theoretically could help accurately inform public opinions and sentiments in times of massive misinformation spread when major world events are occurring, or even during minor public relation events. As one explicit example, this model could be used by social media companies to highlight what they consider as verified information, and flag others as needing more validation or analysis. In another context, public relations teams at companies could train a similar model to recognize their publically available statements, and be used to flag bogus articles that are publishing incorrect statements about their brands  
	- This is of critical importance in today's society where fake or unverified news proliferates like rapid fire through social media, and is extremely influential on public opinion and actions.

2. The other more philosophical benefit of my project is to be able to communicate to users what the reliability and limitations of this type of modelling is. 

	- In reality, we know that large language models are at the forefront of AI research and products, and that there are much more powerful models that are trained on enormous databases and with far greater processing capacity than what was used for this project, that could be trained to detect real or fake news based on their contents with greater success in a wider range of contexts.
    - The importance of conducting this type of more simplified modelling lies in increasing general public understanding of how ML/AI works with regards to these kinds of truth seeking activities. This points to the broader and more important question that faces society on improving the actual reliability of classification models - the truth depends largely on the accuracy and integrity of the human inputs to classifying what is true or false in the data that the machine is learning from.
    - It is important and humbling to acknowledge that the scope of my capstone also depends on the accuracy of the dataset I'm using in classifying news articles as 'fake' or 'real'.  Already reading the comments on the dataset, it appears that the author of the kaggle dataset incorrectly described the classifiers as their opposite - which already presents a problem that needs cleaning, as we will do within the notebook.
    - The intent is to help educate the public and make data science and activities more transparent, which is crucial in an age where people put so much faith into what they see online.  It entails breaking down the expectations of what information ML/AI can give us, as well as holding both corporations and individuals accountable for the type of ML/AI that they are spreading in the world.  This allows the public to stand up to false information and false methods for checking information.


### Data Sources
The dataset used in this report was sourced from Kaggle at the following [link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data).

### Methodology
The analysis was conducted as follows:

1. **Data cleaning and pre-processing**

	- a) Removal of unnecessary columns, duplicate rows, missing data
	- b) Processing raw text from the title and text of each article with the aim of later converting the text into numerical data that can be processed by machine learning algorithms. This included:
 		- i) Tokenizing text which splits the raw text into a vector of separate words and punctuation.
    		- ii) Removal of common stop words and punctuation.
      	 	- iii) Lemmatization of words (reducing words down to their base/root stems to remove some noise).
   	- c) Creating a clean final dataset to be processed by machine learning techniques consisting of:
   		- i) X matrix: contained 2 columns, the the post-processed 'Title' and 'Text' entries  for each article.  Only text or title was used for modelling due to processing constraints, but both were processed to allow flexibility for testing each of the models.
   	 	- ii) y matrix: contained the classification for each article (0 - Real, 1 - Fake).       
	- d)  Splitting the data into train and testing sets using a 75/25 default split.
		- This is done to allow preliminary cross-validation and evaluation of a model.  The model is trained on the training dataset, and then fit to the previously unseen test data.  If the model is not overfit and performs well on unseen data, it is more likely to be able to generalize to new unseen data.

3. **Exploratory data analysis**

	- a) Visualizing distributions of word and character counts for title and text.
 	- b) Summarizing and visualizing most frequently occuring words for each class (Fake, Real).
	- c) Visualizing the distribution of class in the data.
 		- i) The dataset was slightly imbalanced, with an approximate 55/45% split between Real/Fake news.
   	- d) Calculation of a baseline model
		- i) Given that this is a binary classification where the task is to choose between two classes, the baseline model is set to classifying everything as the Majority Class.  This means that the model guesses everything is Real news, which gives 55% accuracy on our existing dataset.  
		- ii) Our machine learning models should have better accuracy than this baseline.

5. **Selecting best metrics to evaluate best performing model**
   
    Since our model is slightly imbalanced, we cannot rely only on high accuracy as a metric of model success. 
    In our dataset, the 'positive' case (1) is 'Fake' news, and 'negative' case (0) is 'Real' news. 
    In this context, there is a high societal cost to both:

     - a) False positives - incorrectly identifying real news as fake (undermines legitimate news sources and spreads distrust).
     - b) False negatives - incorrectly identifying fake news as real (further spreads misinformation and unnecessary hysteria).
    
    As such, the F1 score was chosen as the best indicator of model performance:
    
	- The F1 score is the harmonic mean between precision and recall:
    		- It is calculated as 2 x (Precision x Recall) / (Precision + Recall)
		 
    - Precision is:  	 
	    - "Of all the samples that were labelled as Fake news, how many were actually Fake?"
        	- It is calculated as (True Positives (TP)) / (True Positives (TP) + False Positives (FP))
          	- If there is a high False Positive rate (Real news incorrectly labelled as Fake), precision will be low.
       
    - Recall is:  	 
	    - "Of all actual Fake news in the sample, how many did the model correctly identify?"
        	- It is calculated as (True Positives (TP)) / (True Positives (TP) + False Negatives (FN))
          	- If there is a high False Negative rate (Fake news incorrectly labelled as Real), recall will be low.

    The F1 score can only be high if there is high precision and high recall.
   	- Thus it is a good metric to maximize when the cost of False Negative and False Positives is high, which is true for our problem.
   	- In our case, we consider the best performing model to be the one that has an **F1 score as close as possible to 1**, as this minimizes the risk of both False Positives and False Negatives.      
                
   Another metric that was measured and recorded for future tuning of the models was the PR-AUC value (Precision-Recall Area Under Curve).
   - This shows us the trade-off between precision and recall at different thresholds.
   - If we wanted to do further analysis or model tuning to change the threshold, then a higher PR-AUC score means we will be more likely to maintain a high F1 with a different threshold.
   - The curve also visualizes the what the recall and precision rates are at each threshold, which can be used to pick a threshold that provides a more desirable recall and precision rate.      
        

7. **Modelling using non-neural machine learning methods**

   Four machine learning algorithms that are commonly used for classifying binary data were trained and tested on our news dataset.  
    
   For each model, the method conducted was:

   - 1. Converting the text data to numerical data that can be processed by the algorithm.
        - We used TF-IDF to vectorize the tokenized, lemmatized data from our data cleaning step.  
        - This converts words to numbers based on both their frequency and rarity within a body of text.
   
	- 2. Train and test the model over several iterations with different hyperparameters, with the intent of picking the best performing hyperparameter for that model type.
            - A GridSearch Cross Validation (CV) was run using different parameters for each model.
                - The default cross validation at k-fold of 5 folds.  
                - This meant that the model divided the training dataset into 5 equally sized groups, and performed five iterations of training on 4 out of the 5 sets, while using the remaining group as the test set in each case.
                - In this way, the model picked the best parameters for minimizing the F1 score based on the average performance over these 5 iterations. 
                - This method ensures that the entire dataset is used for training and testing the model, which reduces the likelihood of the model being too specific to the data it is trained on (also known as overfitting), and thus being able to generalize to and perform better on new unseen data.
     
	 - 3. Retrain the model with the best hyperparameter as identified by the GridSearchCV method.
   	 
	 - 4. Calculate evaluation metrics for the best model for that specific type of model.
     
	 - 5. Collate the metrics into a dataframe to compare with the best performing model for all the other model types.

    After collating all metrics, the best performing model was selected based on the one with the **highest f1 score**, per our rationale above.

    The four models and associated hyperparameters selected for this binary classification task are:

	- i) Naive Bayes
	     	- Alpha: (0, 0.0001, 0.001, 0.01, 0.1, 1)
   
	- ii) Decision Tree
		- Minimum Impurity Decrease (0.01, 0.02, 0.03, 0.05)      
		- Criterion ('gini', 'entropy')        
		- Max Depth (1,2,3,4)        
		- Min Samples Split (1,2,3,4)
       
	- iii) Logistic Regression        
		- Fit Intercept (True, False)        
		- Class Weight (None, Balanced)        
		- Penalty (None, L1, L2)
       
	- iv) Support Vector Classification (SVC)          
		- C value (0.1, 1, 100)


9. **Modelling using Recurrent Neural Networking (RNN) methods**

	Based on the task of classifying large bodies of text, the next method was to use RNN models which are more powerful than the non neural methods used above at processing nuanced context of language in news articles.  
    
	The method to complete this was as such:
    
- a) Pre-process text as required for processing in neural networking models
	- i) Tokenize the text using Tensorflow
	- ii) Pad the text so that all samples are the same length
   	- iii) Convert text to numerical vectors using Tensorflow Embedding function
    
- b) Create a Recurrent Neural Network (RNN) model using the Long Short-Term Memory (LSTM) architecture.
	- This was chosen for its ability to complete natural language processing (NLP), in order to detect patterns and nuances in large bodies of text, as required with our dataset. 
        - Due to processor limitations, a fairly simple model base was chosen with:
        	- i) 1 embedding layer to vectorize text data
         	- ii) 1 LSTM hidden layer to analyse patterns
	        - iii) 1 dense output layer with sigmoid activation
		- iv) The model was built with the following parameters:
  			- Optimizing algorithm called 'Adam': This controls how quickly the neural network learns at each step, to help it converge to the correct answer as quickly as possible.  
			- Minimize the binary cross-entropy loss: This a setting commonly used for binary classification task.  It represents the logarithmic distance of the model output from the two classes (0 and 1).  The smaller the distance of each point, the more accurate the model is at correctly predicting the sample's class.
    
- c) Cross-validation and tuning of the model hyperparameters using GridSearchCV
	- Due to processor limitations, only the number of neurons and epochs in the LSTM layer were trialed.

   		-  i) Neurons = (5, 16, 32)
   		-  ii) Epochs = (5,10)

- d) Comparison of tuned RNN model with the non-neural models based on F1 score.

- e) Deep diving the RNN model performance on a specific sample of the dataset using the Lime Explainer package on python. 


### Results

**Recurrent Neural Network Model (RNN)**

This model performed the best compared to the non neural models for correctly classifying news as Fake or Real.  Note that due to the inherent nature of neural networks, the results in the jupyter notebook change slightly each time it is run, however the RNN model still far outperforms the other models each time.

During this run, the best model was the RNN with an F1 score of 0.868.
Its optimal hyperparameters were 10 epochs and 16 neurons. 
On the test data, it achieved 0.872 accuracy, 0.913 recall, 0.827 precision and 0.954 PR-AUC.

Of the non-neural models, Logistic Regression performed the best with an F1 score of 0.775.

Below is a summary of how each F1 score affected the actual False Positives (Real news incorrectly identified as fake) and False Negatives (Fake news incorrectly identified as Real) by each model out of the test dataset of 250 samples:

- RNN model
    - F1 Score: 0.868
    - False Positives: 22
    - False Negatives: 10
    - Total incorrect classifications: 32 (12.8%)

- Logistic Regression model
    - F1 Score: 0.775
    - False Positives: 24
    - False Negatives: 27
    - Total incorrect classifications: 51 (20.4%)
    
- SVC model
    - F1 Score: 0.730
    - False Positives: 12
    - False Negatives: 42
    - Total incorrect classifications: 54 (21.6%)
    
- Naive Bayes model
    - F1 Score: 0.677
    - False Positives: 16
    - False Negatives: 48
    - Total incorrect classifications: 64 (25.6%)
    
- Decision Tree model
    - F1 Score: 0.490
    - False Positives: 2
    - False Negatives: 77
    - Total incorrect classifications: 79 (31.6%)

The optimal hyperparameters and other metric scores for each model are noted below:

**Logistic Regression Model**

The optimal hyperparameters identified were a balanced class weight, intercept of 0, and L2 regularization. This resulted in test data metrics of 0.796 accuracy, 0.765 recall, 0.786 precision, 0.775 f1 score and 0.865 PR-AUC.

**Naive Bayes Model**

The optimal hyperparameter identified was alpha = 0.1,  resulting in test data metrics of 0.744 accuracy, 0.583 recall, 0.807 precision, 0.677 f1 score and 0.826 PR-AUC. 

**Decision Tree Model**

The optimal hyperparameters identified were gini criteria, max depth of 4, minimum impurity decrease of 0.01 and minimum sample split of 2.  This resulted in test data metrics of 0.684 accuracy, 0.330 recall, 0.950 precision, 0.490 f1 score and 0.817 PR-AUC.

**Support Vector Classifier Model**

The optimal hyperparameters identified was a C value of 100, resulting in test data metrics of 0.784 accuracy, 0.635 recall, 0.859 precision, 0.730 f1 score and 0.867 PR-AUC.

**Qualitative Findings**

The limitations of using non-neural networking is that we are processing the text into numbers that only represent their frequency and uniqueness to the text.

The challenge of detecting fake news is that both fake and real news can often contain much of the same content, and be written in similar syntax.  They may only be separate subtle changes in context, order of facts and language (such as DID vs DID NOT), and less commonly these days by improper punctuation or spelling, as well as sensationalist vocabulary.
  
The complexities of assesing subtle contextual differences in large bodies of text, and the nuance between detecting fake and real news is a challenge that is best approached with using more complex recurrent neural network models to learn and interpet subtle patterns that can understand context of words in a sentence or paragraph.  

As expected, the best model for detecting fake news was by far the RNN model over more traditional methods based on its high F1 score. In the context of classifying Fake News, there are high risks to either misclassifying Fake news as Real, and Real news as Fake.  The best metric for striking a balance between correctly classifying both classes is the F1 score.

### Next steps and Recommendations

1. Attempt to improve model performance with the following methods:
   - a) Keeping common stop words and punctuation in the processed data. These could provide important nuances in differing between real and fake publications.  
   - b) Using both title and text as inputs to the model, as opposed to just one category.       
   - c) Further tuning of hyperparameters in the RNN model, such as additional layers, neuron count, batch size, number of epochs etc.
   - d) Use of ensemble methods which combine multiple models to improve results by benefiting from the wisdom of the crowd.
   - e) Using deeply trained word embedding libraries such as Word2Vec during text pre-processing instead of the default Tensorflow/Keras embedder.  This was attempted but took an unrealistic time to process, so was left out. 

These steps were all left out due to limited processing capacity within the time constraints given.
    
2. Use pre-trained language models such as BERT to process and train the data. 
There are many pre-existing models that have been trained extensively to recognize language patterns, such as the BERT model designed and maintained by google.  It would be hugely beneficial to use this kind of pre-trained neural network that has benefited from vast amounts of data and tuning to more successfully identify patterns in our dataset.   Further tuning can then occur.
This was also attempted during the project but dropped due to computer processing constraints - the computing memory available was unable to process these larger models in the jupyter notebook environment.

3. Deploy the model in a business setting including but not limited to:
    - a) Creating an app where users can submit the article contents or a link to article, and have the article assess as fake or real
    - b) Embedding the model onto social media platforms to automatically flag news as fake or real.
    
4. Create further high level summaries of the modelling to use as educational tools for the general public regarding general principles of how AI/ML is used to process and analyse text.


### Outline of project

- [Link to Final Technical Report](https://github.com/mpacielim/FakeNewsCapstoneFinal/blob/976824efafc3056b2466e10a85fd07755fe2f589/Fake%20News%20Capstone%20Technical%20Report.ipynb)
- [Link to Raw Fake News Dataset](https://github.com/mpacielim/FakeNewsCapstoneFinal/tree/main/data)
- [Link to Raw Fake News Dataset Original Location in Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)
- Note that the dataset linked above was above the maximum 100 MB file size upload allowed by github, hence the [github large file storage (LFS) system](https://git-lfs.com/) was used.
	- The .gitattributes file in this repository was required to upload the dataset through the github LFS platform.


##### Contact and Further Information

Melissa Paciepnik
[LinkedIn](https://www.linkedin.com/in/melissa-paciepnik-43a85979/)
