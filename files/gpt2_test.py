# https://colab.research.google.com/drive/1vnpMoZoenRrWeaxMyfYK4DDbtlBu-M8V?usp=sharing#scrollTo=H1ag9Z0iZbzG
# https://github.com/ivanlai/Conditional_Text_Generation
# https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d

from gpt2_train import SPECIAL_TOKENS, get_tokenizer, get_model, MAXLEN
import torch
import nltk

tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='/files/gpt2-data/checkpoint-465/pytorch_model.bin') 


title = "Artificial intelligence in Radiology."

prompt = SPECIAL_TOKENS['bos_token'] + title +  SPECIAL_TOKENS['sep_token']



### METHOD 1   
# device = torch.device("cuda")
# inputs = tokenizer(prompt, max_length=MAXLEN, truncation=True, return_tensors="pt").to(device)
# output = model.generate(**inputs,  max_length=MAXLEN )
# decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
# print(decoded_output)



### METHOD 2      
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cuda")
generated = generated.to(device)
model.eval()

# Top-p (nucleus) text generation (10 samples):
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



### OR
# Beam-search text generation:
sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                max_length=MAXLEN,                                                      
                                num_beams=5,
                                repetition_penalty=5.0,
                                early_stopping=True,      
                                num_return_sequences=1
                                )

for i, sample_output in enumerate(sample_outputs):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    print("{}: {}\n\n".format(i+1,  text))