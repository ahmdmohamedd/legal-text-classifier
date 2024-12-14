# **Legal Text Classifier**

This project focuses on classifying legal documents into categories based on their content. The system uses text preprocessing, cleaning, and feature extraction techniques, along with machine learning models (Logistic Regression and Random Forest) to categorize the outcomes of legal cases. The project demonstrates an efficient and reliable approach to document classification using Python and popular machine learning libraries.

---

## **Table of Contents**

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Techniques and Models](#techniques-and-models)  
- [Getting Started](#getting-started)  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [How to Run the Code](#how-to-run-the-code)  
- [Results](#results)  
- [Future Improvements](#future-improvements)  
- [Contributing](#contributing)  

---

## **Project Overview**

This project aims to classify legal documents based on their text content. It leverages text preprocessing and machine learning techniques to categorize cases accurately. The core objective is to automate the categorization of legal cases into relevant classes, which can aid in quick retrieval, search, and analysis of legal outcomes.

By employing logistic regression and Random Forest classifiers, the system extracts meaningful patterns from the text data, ensuring reliable classification performance. This project highlights text preprocessing, feature extraction, and the application of machine learning models for text classification tasks.

---

## **Dataset**

- The dataset contains legal case documents.
- It includes columns such as `case_id`, `case_outcome`, `case_title`, and `case_text`.
- This dataset is sourced from Kaggle and is designed for the classification of legal documents into predefined categories.
- The dataset ensures diversity in case titles and outcomes, providing ample training and testing examples to build a robust classifier.

**Kaggle Dataset:** [https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset](https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset)

---

## **Techniques and Models Used**

- **Text Preprocessing & Cleaning**:  
  - Lowercasing text  
  - Removing special characters, numbers, and unnecessary whitespace  
  - Tokenization and basic text normalization  

- **Feature Extraction**:  
  - TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization  

- **Machine Learning Models**:  
  - **Logistic Regression**  
  - **Random Forest Classifier**  

These models were selected for their efficiency in text classification tasks, ensuring quick training and high performance on document categorization.

---

## **Getting Started**

This section will help you set up and run the project on your local machine.

### **Prerequisites**

Before you run the project, ensure you have:

- Python 3.x installed  
- Jupyter Notebook or JupyterLab  
- Pandas  
- Scikit-learn  
- Numpy  
- Other essential Python libraries  

You can install these dependencies using the following:

```bash
pip install pandas scikit-learn jupyter
```

---

## **Installation**

Clone the repository:

```bash
git clone https://github.com/ahmdmohamedd/legal-text-classifier.git
```

Navigate to the project directory:

```bash
cd legal-text-classifier
```

---

## **Project Structure**

Here's a breakdown of the project structure:

```
legal-text-classifier/
├── README.md
└── legal_text_classifier.ipynb
```
 
- **`legal_text_classifier.ipynb`**: Jupyter Notebook demonstrating the end-to-end process of the classifier  
- **`README.md`**: Documentation explaining the project setup and implementation  

---

## **How to Run the Code**

1. Open the Jupyter Notebook by running:

```bash
jupyter notebook
```

2. Launch the `legal_text_classifier.ipynb` notebook to explore and execute the workflow.
3. Follow the steps in the notebook to:
   - Load and clean the data
   - Preprocess text data  
   - Train Logistic Regression and Random Forest models  
   - Evaluate classifier performance and accuracy  

---

## **Results**

The classifier achieves high accuracy in categorizing legal documents. The comparison of Logistic Regression and Random Forest models shows that both classifiers perform efficiently, with Random Forest often yielding more robust results. The system provides reliable classification outcomes that demonstrate the feasibility of automating legal case document categorization.

---

## **Future Improvements**

- **Deep Learning Integration**: Incorporate advanced models like BERT or transformers for better text embeddings.  
- **Hyperparameter Tuning**: Optimize hyperparameters to improve performance across classifiers.  
- **Scalability**: Implement the system for larger datasets with more diverse case categories.  
- **Visualization Tools**: Add visualizations to explore text patterns and trends in case categories.  

---

## **Contributing**

If you'd like to contribute to this project, feel free to fork this repository and submit a pull request. Whether it's bug fixes, enhancements, or suggestions, your contributions are highly appreciated.

---
