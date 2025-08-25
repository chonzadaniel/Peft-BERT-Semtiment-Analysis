# ------------------------------------------ Installation of frameworks -----------------------------------------------
!pip install --quiet bitsandbytes
!pip install --quiet --upgrade transformers # Install latest version of transformers
!pip install --quiet --upgrade accelerate
!pip install --quiet sentencepiece
!pip install transformers bitsandbytes accelerate --quiet

# ----------------------------------------------- Import Library -----------------------------------------------------
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging


# ------------------------------------------------------- HuggingFace API Environment Setup ----------------------------------------------------
os.environ["HUGGINGFACE_API_KEY"] = open('HUGGINGFACE_API_TOKEN.txt','r').read()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------- Load Dataset --------------------------------------------------------------------
dataset = load_from_disk("/Users/emmanueldanielchonza/Documents/Parameter-Efficient-Fine-tuning-LLMs/data")

# ---------------------------------------------------------- Map LABEL2ID and ID2LABEL ------------------------------------------------------
LABEL2ID = {'positive': 1, 'negative': 0}
ID2LABEL = {1: 'positive', 0: 'negative'}

# ------------------------------------------------------ Merge Classification LoRA Adapter into Base BERT Model -----------------------------------
peft_model_id = "qlora-bert-sentiment-adapter"
config = PeftConfig.from_pretrained(save_path) # peft_model_id or save_path

# ------------------------------------------------------- Configure the base model ------------------------------------------------------
# Get the model and instantiate tokenizer
model_checkpoint = "google-bert/bert-base-uncased"

logging.set_verbosity_error()  # suppress info/progress messages

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                id2label=ID2LABEL,
                                                                label2id=LABEL2ID,
                                                                num_labels=2)

# --------------------------------------------------------- Create the tokenizer -----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, fast=True)

peft_model = PeftModel.from_pretrained(base_model, save_path).to('cuda')

# ----------------------------------------------- Create the merged model ------------------------------------------
merged_cls_model = peft_model.merge_and_unload()

# ---------------------------------------------Save the merged model ---------------------------------------------
save_path = 'merged-qlora-bert-classifier'

merged_cls_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# ------------------------------------------- Load the Merged Model --------------------------------------------
# load the merged BERT model 
cls_model = AutoModelForSequenceClassification.from_pretrained('merged-qlora-bert-classifier',
                                                                id2label=ID2LABEL,
                                                                label2id=LABEL2ID,
                                                                num_labels=2)
# Use the merged model to create tokenizer
tokenizer = AutoTokenizer.from_pretrained('merged-qlora-bert-classifier', fast=True)


#---------------------------------------------- Instantiate the classifier --------------------------------------
clf = pipeline(task='text-classification', 
               model=cls_model, 
               tokenizer=tokenizer, 
               device='cuda')

# ------------------------------------------- Use the Classifier to Make Predictions ----------------------------------
document = "The movie was not good at all"
clf(document)

document1 = "The movie was amazing"
clf(document1)