# ------------------------------------------ Installation of frameworks -----------------------------------------------
!pip install --quiet bitsandbytes
!pip install --quiet --upgrade transformers # Install latest version of transformers
!pip install --quiet --upgrade accelerate
!pip install --quiet sentencepiece
!pip install transformers bitsandbytes accelerate --quiet

# ----------------------------------------------- Import Library -----------------------------------------------------
import os
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# ------------------------------------------------------- HuggingFace API Environment Setup ----------------------------------------------------
os.environ["HUGGINGFACE_API_KEY"] = open('HUGGINGFACE_API_TOKEN.txt','r').read()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------- Load Dataset --------------------------------------------------------------------
dataset = load_from_disk("/Users/emmanueldanielchonza/Documents/Parameter-Efficient-Fine-tuning-LLMs/data")

# ---------------------------------------------------------- Map LABEL2ID and ID2LABEL ------------------------------------------------------
LABEL2ID = {'positive': 1, 'negative': 0}
ID2LABEL = {1: 'positive', 0: 'negative'}

# -------------------------------------------- Load the base BERT model first ------------------------------------
cls_model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased',
                                                                id2label=ID2LABEL,
                                                                label2id=LABEL2ID,
                                                                num_labels=2)

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', fast=True)

# ----------------------------------------- Load the fine-tuned BERT model ------------------------------------------
cls_model.load_adapter(peft_model_id='qlora-bert-sentiment-adapter',
                       adapter_name='sentiment-classifier')

# --------------------------------- Use the Fine-tuned model for Classification ------------------------------
# Locally trained \ saved model
clf = pipeline(task='text-classification', 
               model=cls_model, 
               tokenizer=tokenizer, 
               device='cuda')

document = "The movie was not good at all"

clf(document)

# --------------------------------------- Fine-tuned Transformer performance on Test Data -----------------------------
dataset['test'][:2]

# ------------------------------------------- Make Predictions on Test Set ------------------------------------------
%%time
# Instantiate the predictor
predictions = clf(dataset['test']['review'],
                  batch_size=512, 
                  max_length=512, 
                  truncation=True)
# Get predictions
predictions = [pred['label'] for pred in predictions]

# Get prediction Labels
predictions = [0 if item == 'NEGATIVE' else 1 for item in predictions]
labels = dataset['test']['label']

# Evaluate the prediction on test set
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(labels, predictions))
pd.DataFrame(confusion_matrix(labels, predictions))


