import gradio as gr
from transformers import pipeline

import re

pipe = pipeline("summarization", model="kkasiviswanath/bart_summarizer_deploy_v1")

def summarize_email(email_body, pipe):
    # Tokenize the input text
    input_tokens = pipe.tokenizer(email_body, return_tensors='pt', truncation=False)
    input_length = input_tokens['input_ids'].shape[1]

    # Adjust max_length to be a certain percentage of the input length
    adjusted_max_length = max(3, int(input_length * 0.6))  # Ensure a minimum length
    # Generate summary with dynamic max_length
    gen_kwargs = {
        "length_penalty": 2.0,
        "num_beams": 5,
        "max_length": adjusted_max_length,
        "min_length": 3
    }

    summary = pipe(email_body, **gen_kwargs)[0]['summary_text']
    return summary

# Generate summaries for the test dataset
def generate_summary(text):
    email_body = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text).strip())
    summary = summarize_email(email_body, pipe)
    return summary

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=generate_summary, inputs="text", outputs="text")
demo.launch(share=True)