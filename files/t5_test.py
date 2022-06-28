from transformers import pipeline, AutoModel
from transformers import GPT2Tokenizer, GPT2Model
import torch
from process_data import get_dataset
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import transformers
from datasets import load_dataset, load_metric
import numpy as np


model_name = "t5-base-abstract-generation/checkpoint-2400"
model_dir = model_name

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 100
max_target_length = 512

text = """Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study"""

inputs = ["abstractify: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=max_target_length)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
print(decoded_output)
predicted_title = nltk.sent_tokenize(decoded_output[0].strip())

print(predicted_title)



################

# # classifier = pipeline("text-generation", model="cahya/abstract-generator")

# # result = classifier("Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study")

# # print(result)

# # device = torch.device("cuda")




# MODEL_NAME = "healx/gpt-2-pubmed-medium"

# tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# model = GPT2Model.from_pretrained(
#     MODEL_NAME
# )


# generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

# result = generator()

# print(result)

# inputs = tokenizer(, return_tensors="pt")

# outputs = model(**inputs)

# # print(outputs)\
# PROMPT = "Deep learning for lung cancer prognostication A retrospective multi-cohort radiomics study"

# generated = torch.tensor(tokenizer.encode(PROMPT)).unsqueeze(0)
# # generated = generated.to(device)

# print(generated)

# sample_outputs = model.generate(
#                                 generated, 
#                                 #bos_token_id=random.randint(1,30000),
#                                 do_sample=True,   
#                                 top_k=50, 
#                                 max_length = 300,
#                                 top_p=0.95, 
#                                 num_return_sequences=3
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))



# text = """Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study"""

# inputs = tokenizer(text, max_length=200, truncation=True, return_tensors="pt")
# output = model.generate(**inputs, min_length=10, max_length=3000)
# decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
# predicted_title = nltk.sent_tokenize(decoded_output.strip())
# print(predicted_title)