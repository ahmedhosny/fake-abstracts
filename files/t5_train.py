# https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887


from process_data import get_dataset
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import transformers
from datasets import load_dataset, load_metric
import numpy as np

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

ds = get_dataset()

max_input_length = 100
max_target_length = 512
prefix = "summarize: "

# remove small items.
ds = ds.filter(
    lambda example: 
        (len(example['title']) >= 20)
        and
        (len(example['abstract']) >= 500) 
)

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples):
    # TITLES
    inputs = [prefix + text for text in examples["title"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # ABSTRACTS
    with tokenizer.as_target_tokenizer():
        texts_cleaned = [clean_text(text) for text in examples["abstract"]]
        labels = tokenizer(texts_cleaned, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_ds = ds.map(preprocess_data, batched=True)

print(tokenized_ds)

batch_size = 16
model_name = "t5-base-abstract-generation"
model_dir = model_name

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    # report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

metric = load_metric("rouge")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()