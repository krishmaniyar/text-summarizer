# Text Summarizer - End-to-End ML Project

An end-to-end text summarization application using fine-tuned Pegasus transformer model. This project implements a complete machine learning pipeline from data ingestion to model deployment, with a user-friendly web interface for generating summaries.

## 🚀 Features

- **End-to-End ML Pipeline**: Complete workflow from data ingestion to model evaluation
- **Fine-tuned Pegasus Model**: Pre-trained on CNN/DailyMail and fine-tuned on SAMSum dataset
- **Web Interface**: FastAPI-based web application with a modern UI
- **RESTful API**: Easy-to-use API endpoints for text summarization
- **Model Training**: Automated training pipeline with configurable parameters
- **Model Evaluation**: Built-in evaluation metrics (ROUGE, BLEU scores)

## 🛠️ Tech Stack

- **Framework**: FastAPI, Uvicorn
- **ML Library**: Transformers (Hugging Face), PyTorch
- **Data Processing**: Datasets, Pandas, NLTK
- **Evaluation**: ROUGE Score, SacreBLEU
- **Frontend**: Jinja2 Templates, HTML/CSS
- **Configuration**: PyYAML, Python-Box

## 📁 Project Structure

```
text-summarizer/
├── artifacts/                 # Generated artifacts (models, datasets, metrics)
│   ├── data_ingestion/        # Raw and processed datasets
│   ├── data_transformation/   # Tokenized datasets
│   ├── model_trainer/         # Trained model checkpoints
│   └── model_evaluation/      # Evaluation metrics
├── config/                    # Configuration files
│   └── config.yaml           # Main configuration
├── research/                  # Jupyter notebooks for experimentation
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   └── 05_Model_evaluation.ipynb
├── src/                       # Source code
│   └── text_summarizer/
│       ├── components/        # Core components
│       ├── config/           # Configuration management
│       ├── pipeline/          # Training and prediction pipelines
│       └── utils/            # Utility functions
├── static/                    # Static files (CSS, JS)
├── templates/                 # HTML templates
├── app.py                     # FastAPI application
├── main.py                    # Training pipeline entry point
├── requirements.txt           # Python dependencies
└── params.yaml               # Training hyperparameters
```

## 📋 Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- Git

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/krishmaniyar/text_summarizer.git
cd text-summarizer
```

### Step 2: Create a Conda Environment

```bash
conda create -n summary python=3.8 -y
conda activate summary
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- transformers
- torch
- fastapi
- uvicorn
- datasets
- rouge_score
- and more...

## 🎯 Usage

### Running the Web Application

1. **Start the FastAPI server**:

```bash
python app.py
```

2. **Access the application**:

Open your browser and navigate to:
```
http://localhost:8080
```

3. **Generate Summaries**:

- Enter your text in the input field
- Click the "Summarize" button
- View the generated summary

### API Endpoints

#### 1. Home Page
- **URL**: `GET /`
- **Description**: Returns the main web interface

#### 2. Generate Summary
- **URL**: `POST /predict`
- **Description**: Generates a summary for the input text
- **Parameters**:
  - `text` (form-data): The text to summarize
- **Response**: HTML page with the summary

#### 3. Train Model
- **URL**: `GET /train`
- **Description**: Triggers the training pipeline
- **Response**: JSON with training status

### Example API Usage

Using `curl`:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Your long text here that needs to be summarized..."
```

Using Python:

```python
import requests

url = "http://localhost:8080/predict"
data = {"text": "Your long text here..."}
response = requests.post(url, data=data)
print(response.text)
```

## 🏋️ Training the Model

### Running the Training Pipeline

To train the model from scratch, run:

```bash
python main.py
```

This will execute the complete pipeline:

1. **Data Ingestion**: Downloads and extracts the SAMSum dataset
2. **Data Validation**: Validates the dataset structure
3. **Data Transformation**: Tokenizes the data using Pegasus tokenizer
4. **Model Training**: Fine-tunes the Pegasus model on the dataset
5. **Model Evaluation**: Evaluates the model and generates metrics

### Training Configuration

Edit `params.yaml` to adjust training hyperparameters:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
```

### Model Configuration

Edit `config/config.yaml` to modify:
- Data paths
- Model checkpoints
- Tokenizer settings
- Evaluation metrics

## 📊 Model Details

- **Base Model**: `google/pegasus-cnn_dailymail`
- **Fine-tuned Dataset**: SAMSum (conversation summarization)
- **Tokenizer**: Pegasus tokenizer
- **Generation Parameters**:
  - Length penalty: 0.8
  - Number of beams: 8
  - Max length: 128 tokens

## 🔬 Research Notebooks

The `research/` directory contains Jupyter notebooks for each stage of the pipeline:

- `01_data_ingestion.ipynb`: Data download and preprocessing
- `02_data_validation.ipynb`: Data quality checks
- `03_data_transformation.ipynb`: Tokenization and feature engineering
- `04_model_trainer.ipynb`: Model training experiments
- `05_Model_evaluation.ipynb`: Model evaluation and metrics

## 📝 Configuration Files

### config.yaml
Main configuration file containing:
- Artifact root directories
- Data ingestion settings
- Model paths
- Evaluation settings

### params.yaml
Training hyperparameters and model arguments.

## 🐳 Docker Support

A `Dockerfile` is included for containerized deployment. Build and run:

```bash
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👤 Author

**Krish Maniyar**

- Email: krishmaniyar27@gmail.com
- GitHub: [@krishmaniyar](https://github.com/krishmaniyar)

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Google Research for the Pegasus model
- SAMSum dataset creators

## 📚 Additional Resources

- [Pegasus Paper](https://arxiv.org/abs/1912.08777)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Note**: This project is based on the end-to-end ML project structure. Make sure to have sufficient computational resources (GPU recommended) for training the model.
