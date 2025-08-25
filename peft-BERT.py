# ----------------------------------------------------------- Freeup Environment ----------------------------------------------------------
import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType, replace_lora_weights_loftq


# ------------------------------------------------------- Clear Local Disk ----------------------------------------------------------------------
torch.cuda.empty_cache()

# ------------------------------------------------------- HuggingFace API Environment Setup ----------------------------------------------------
os.environ["HUGGINGFACE_API_KEY"] = open('HUGGING_API_TOKEN.txt','r').read()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------- Load Dataset ------------------------------------------------------------
dataset = load_from_disk("/Users/emmanueldanielchonza/Documents/Parameter-Efficient-Fine-tuning-LLMs/data")

# ----------------------------------------------------------------- Setup the Evaluation Metrics --------------------------------------------
## Load the specific model performance evaluation metrics
metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("f1")
metric4 = evaluate.load("accuracy")

## Create a performance evaluation function
def evaluate_performance(predictions, references):
    precision = metric1.compute(predictions=predictions, references=references, average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=references, average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=references, average="macro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=references)["accuracy"]
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

## Calculate the evaluation scores
predictions = [1,0,1,1,0]
references = [1,1,0,1,0]
scores = evaluate_performance(
    predictions=predictions, references=references
)                                                       # Apply the evaluation function

# ----------------------------------------------------------- Preprocessing the data -----------------------------------------
model_checkpoint = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Tokenization example
# tokenizer("Hello, this is a sentence!")

## Create a function to tokenize dataset
def preprocess_function(examples):
    # max length is 512 as that is the context window limit of BERT models
    # It can process documents of upto 512 tokens each input
    model_inputs = tokenizer(examples['review'], max_length=512, truncation=True)
    model_inputs["label"] = examples["sentiment"]
    return model_inputs

# Example application
# preprocess_function(imdb_data["train"][:2])

tokenized_datasets = imdb_data.map(preprocess_function, batched=True)

# remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns('review')
tokenized_datasets = tokenized_datasets.remove_columns('sentiment')

# ----------------------------------------- Parameter Efficient Fine-tuning the Transformer Model ------------------------------
## Model Quantization and Configuration
### Model Qunatization
config = BitsAndBytesConfig(
    load_in_4bit=True, # quantize the model to 4-bits when you load it
    bnb_4bit_quant_type="nf4", # use a special 4-bit data type for weights initialized from a normal distribution
    bnb_4bit_use_double_quant=True, # nested quantization scheme to quantize the already quantized weights
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 for faster computation
    llm_int8_skip_modules=["classifier", "pre_classifier"]
)

### Model Configuration
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                           id2label=ID2LABEL,
                                                           label2id=LABEL2ID,
                                                           num_labels=2,
                                                           quantization_config=config)

### Peft Model
model = prepare_model_for_kbit_training(model)

### Gets the trainable parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

### print the model
# print_trainable_parameters(model)

### Model architecture
# model

# Set up the LoRA configuration for the model
config = LoraConfig(
    r=8,  # Rank of the LoRA matrices; a smaller rank reduces memory usage but may affect model performance.
    lora_alpha=32,  # Scaling factor applied to the LoRA updates; helps control the contribution of the LoRA weights.
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # Specify the modules (weight matrix) within the model where LoRA is applied.
    lora_dropout=0.05,  # Dropout probability for LoRA layers to prevent overfitting during training.
    bias="none",  # Specifies whether to add learnable biases to the LoRA layers.
    task_type=TaskType.SEQ_CLS  # Defines the task type, here it's set to sequence classification.
)

# Apply the LoRA configuration to the model
peft_model = get_peft_model(model, config)

# Print the number of trainable parameters in the model after applying LoRA
# print_trainable_parameters(peft_model)

## Send the model to device
peft_model.device

# if batch size is 64
# if total documents are 8000
# total number of steps (batches of data) to complete 1 full epoch is?
8000 // 32

# total steps to run two epochs are?
250 * 2


batch_size = 32
metric_name = "f1"

# ---------------------------------------------- Set up the training arguments -------------------------------------
args = TrainingArguments(
    output_dir="bert-cls-qlorafinetune-runs",  # Directory where the model checkpoints and outputs will be saved.
    eval_strategy="steps",                          # Perform evaluation at regular intervals during training.
    save_strategy="steps",                          # Save the model checkpoint at regular intervals.
    learning_rate=1e-4,                             # Initial learning rate for the optimizer.
    logging_steps=20,                               # Log training metrics every 20 steps.
    eval_steps=20,                                  # Perform evaluation every 20 steps.
    save_steps=50,                                  # Save the model checkpoint every 50 steps.
    per_device_train_batch_size=batch_size,         # Batch size per GPU/TPU core/CPU during training.
    per_device_eval_batch_size=batch_size,          # Batch size per GPU/TPU core/CPU during evaluation.
    max_steps=250,                                  # Stop training after 250 total steps.
    weight_decay=0.01,                              # Apply weight decay to reduce overfitting.
    metric_for_best_model=metric_name,              # Metric to use for selecting the best model during evaluation.
    push_to_hub=False,                              # Do not push the model to the Hugging Face Hub after training.
    fp16=True,                                      # Use 16-bit floating point precision to reduce memory usage and speed up training.
    optim="paged_adamw_8bit",                       # Use an 8-bit AdamW optimizer for memory efficiency and faster computation.
)

# ---------------------------------------------------- Setup Datacollator --------------------------------------
# Sets up padding to the longest tokens
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --------------------------------------------------- Setup the Compute function ---------------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return evaluate_performance(predictions=predictions, references=labels)

# --------------------------------------------------- Create the Trainer -----------------------------------------
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ------------------------------------------------- Train the Model -------------------------------------------
trainer.train()

# ------------------------------------------------ Save the trained model ---------------------------------------
save_path = 'qlora-bert-sentiment-adapter'
trainer.save_model(save_path)

# --------------------------------------- Clear the Training Logs after Saving model ------------------------------
# remove model checkpoints
!rm -rf bert-cls-qlorafinetune-runs

# ---------------------------------------------- Check the saved model size ------------------------------------
!du -sh * | sort -hr | grep qlora


