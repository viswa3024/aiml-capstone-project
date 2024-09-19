import gradio as gr
from transformers import pipeline

import re

qa_pipeline = pipeline('text2text-generation', model='kkasiviswanath/bart_large_qna_dataset_2_v1',
                       tokenizer='kkasiviswanath/bart_large_qna_dataset_2_v1')

generation_kwargs = {
    'num_beams': 5,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'forced_bos_token_id': 0,
    'forced_eos_token_id': 2,
    'temperature':1.0,
    'top_p':0.9,
    'do_sample': True,  # Enable sampling
    'max_length': 100  # You can also set other parameters like max_length here
}


def generate_answer(question):
    result = qa_pipeline(question, **generation_kwargs)
    response = result[0]['generated_text']
    return response

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=generate_answer, inputs="text", outputs="text")
demo.launch(share=True)