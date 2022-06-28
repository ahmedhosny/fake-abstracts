from gpt2_train import get_tokenizer, get_model, MAXLEN
import torch
from transformers import GPT2Tokenizer, GPT2Model

device = torch.device("cuda")

MODEL_NAME = "healx/gpt-2-pubmed-large"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

model = GPT2Model.from_pretrained(
    MODEL_NAME
)


prompt = "Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study"
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)
model.to(device)

# Top-p (nucleus) text generation:
sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                min_length=50, 
                                max_length=MAXLEN,
                                top_k=30,                                 
                                top_p=0.7,        
                                temperature=0.9,
                                repetition_penalty=2.0,
                                num_return_sequences=2
                                )

for i, sample_output in enumerate(sample_outputs):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    print("{}: {}\n\n".format(i+1,  text))



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





# tokenizer = get_tokenizer()
# model = get_model(tokenizer)


# prompt = "Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study"

# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# device = torch.device("cuda")
# generated = generated.to(device)

# model.eval()
# sample_outputs = model.generate(generated, 
#                                 do_sample=True,   
#                                 max_length=MAXLEN,                                                      
#                                 num_beams=5,
#                                 repetition_penalty=5.0,
#                                 early_stopping=True,      
#                                 num_return_sequences=1
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))