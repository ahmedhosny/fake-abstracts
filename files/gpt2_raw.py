# https://colab.research.google.com/drive/1vnpMoZoenRrWeaxMyfYK4DDbtlBu-M8V?usp=sharing#scrollTo=H1ag9Z0iZbzG
# https://github.com/ivanlai/Conditional_Text_Generation
# https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d

from gpt2_train import get_tokenizer, get_model, MAXLEN
import torch

tokenizer = get_tokenizer()
model = get_model(tokenizer)


prompt = "Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study"

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cuda")
generated = generated.to(device)

model.eval()
sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                max_length=MAXLEN,                                                      
                                num_beams=5,
                                repetition_penalty=5.0,
                                early_stopping=True,      
                                num_return_sequences=1
                                )

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))