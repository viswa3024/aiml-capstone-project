import gradio as gr
from transformers import pipeline

import re

#email_subject_pipeline = pipeline("text-generation", model="kkasiviswanath/distilgpt2_email_subject_summarizer")
email_subject_pipeline = pipeline("text-generation", model="kkasiviswanath/gpt2_medium_email_subject_summarizer_v1")

def clean_subject(response):
    print(response)
    lst = response.split('<|sep|>')
    if (len(lst) >= 2):
        response = lst[1].replace("<|endoftext|>","")
    return response


def generate_subject(email: str):
    prompt = f"<|startoftext|> {email} <|sep|>"

    # Use the pipeline to generate the text
    sample_outputs = email_subject_pipeline(prompt, max_new_tokens=12, num_beams=5, early_stopping=True, num_return_sequences=1)

    # The pipeline output is a list of dictionaries, so extract the generated text
    subject = clean_subject(sample_outputs[0]['generated_text'])
    return subject



#https://huggingface.co/spaces/gradio/theme-gallery
#theme = gr.themes.ThemeClass.from_hub("freddyaboulton/dracula_revamped")
#upsatwal/mlsc_tiet
#gradio/seafoam
theme = gr.themes.ThemeClass.from_hub("upsatwal/mlsc_tiet")


css = """
.gradio-container-4-41-0 .md pre {
    background: #374151 !important;
}
code {
    color: #FFFFFF;
    padding: 5px;
    border-radius: 5px;
    font-size: 14px;
    white-space: pre-wrap;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 25px;display: flex;justify-content: center;height:100px;">
            <img src="https://raw.githubusercontent.com/viswa3024/aiml-capstone-project-email/main/logo.png" alt="Logo" height="100px" width="50%">
        </div>
        """
    )
     
    gr.Markdown("# AI-based Generative QA System")
    with gr.Tab("Email Subject Line Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ## Group 17
                        ```
                        Chandrasekhar B
                        K Kasi Viswanath
                        ```
                        ## Model
                        ```
                        Fine tuned gpt2-medium

                        ```
                        ## Training Dataset
                        ```
                        Annotated Enron Subject Line Corpus(AESLC) dataset

                        ```

                        """)
            with gr.Column(scale=2):
                text_input_email = gr.Textbox(lines=8, label="Email")
                text_output_subject = gr.Textbox(label="Generated Subject")
            
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    """)
            with gr.Column(scale=2):
                btn_generate_subject = gr.Button("Generate Subject")
    with gr.Tab("Question Answering on AIML Queries"):
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
                            ## Group 17
                            ```
                            Chandrasekhar B
                            K Kasi Viswanath
                            ```
                            ## Model
                            ```
                            Fine tuned gpt2-medium

                            ```
                            ## Training Dataset
                            ```
                            Custom
                            ```

                            """)
                with gr.Column(scale=2):
                    text_input_question = gr.Textbox(label="Question")
                    text_output_answer = gr.Textbox(lines=8, label="Generated Answer")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        """)
                with gr.Column(scale=2):
                    btn_generated_answer = gr.Button("Generate Answer")
    btn_generate_subject.click(generate_subject, inputs=text_input_email, outputs=text_output_subject)
    btn_generated_answer.click(generate_subject, inputs=text_input_question, outputs=text_output_answer)
    
demo.launch(share=True)