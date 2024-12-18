# Project Title
Classifying Code Based on Time Complexity Using CodeBERT

## Project Objective
This project aims to classify code snippets into categories based on their time complexity. By leveraging the CodeBERT model, we analyze the structure and logic of code to predict its computational complexity effectively.

## Key Features
- **Dataset**: A curated dataset of code snippets annotated with time complexity labels.
- **Model**: Fine-tuned CodeBERT for the task of code classification.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score.

## Methodology
1. **Data Preparation**: The dataset is preprocessed to tokenize code snippets and prepare them for input into the CodeBERT model.
2. **Model Training**: CodeBERT is fine-tuned using a supervised learning approach on the labeled dataset.
3. **Evaluation**: The trained model is evaluated on a test set to measure its performance using standard metrics.

## Results
- The model achieved a high accuracy in classifying code snippets into their respective time complexity categories.
- Precision, recall, and F1-score metrics were used to validate the robustness of the model.

## Setup Instructions
### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Required libraries: `transformers`, `torch`, `numpy`, `pandas`, `scikit-learn`

### Steps to Run
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the training notebook:
   ```bash
   jupyter notebook codebert_train.ipynb
   ```
   - Follow the steps to preprocess the dataset and train the model.
4. Evaluate the results by running:
   ```bash
   jupyter notebook codebert_results.ipynb
   ```

## Directory Structure
```
project_root/
|-- codebert_train.ipynb      # Training notebook
|-- codebert_results.ipynb    # Results and evaluation notebook
|-- data/                     # Dataset directory
|-- models/                   # Trained models
|-- requirements.txt          # Required Python packages
```

## Future Work
- Extend the model to classify additional algorithmic complexities.
- Explore alternative pre-trained models like GraphCodeBERT for improved performance.
- Integrate real-world datasets to enhance model generalization.

## Contributors
- [Your Name] - Project Lead and Developer

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments
- Hugging Face for the CodeBERT model.
- OpenAI and the developer community for their support.
