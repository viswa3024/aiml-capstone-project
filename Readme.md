![alt text](logo.png)

# AI-based Generative QA System

### Project Overview: Email Subject Line Generation

This project aims to generate email subject lines by leveraging generative models in NLP. Unlike tasks such as news summarization or headline generation, this task requires creating extremely short and concise summaries in the form of email subjects. Here are the key points:

1. **Unique Challenge**: The task involves identifying the most salient sentences from an email body and abstracting the message into a few words.
2. **Dataset**: The Annotated Enron Subject Line Corpus (AESLC) is used, which consists of cleaned, filtered, and deduplicated emails from the Enron Email Corpus. The dataset includes multiple annotated subject lines for better evaluation.
3. **Splits and Statistics**:
   - Train / dev / test split sizes: 14,436 / 1,960 / 1,906
   - Average email length: 75 words
   - Average subject length: 4 words
4. **Generative Models**: The project explores the use of various GPT-2 variants and other models like BART and T5 for generating subject lines.
5. **Evaluation Metrics**: The effectiveness of the generated subject lines is evaluated using metrics such as ROUGE-1, ROUGE-2, and ROUGE-L.

### ROUGE Scores:

| Model            | ROUGE-1                | ROUGE-2                | ROUGE-L                |
|------------------|------------------------|------------------------|------------------------|
| GPT-2            | 0.1409                 | 0.0353                 | 0.1361                 |
| BART Base        | 0.3738                 | 0.2444                 | 0.3665                 |
| BART Large-CNN   | 0.3257                 | 0.2800                 | 0.3257                 |
| T5 Small         | 0.1835                 | 0.0866                 | 0.1732                 |
| T5 Base          | 0.0985                 | 0.0353                 | 0.0959                 |



| Model            | ROUGE-1 (Recall) | ROUGE-1 (Precision) | ROUGE-1 (F1) | ROUGE-2 (Recall) | ROUGE-2 (Precision) | ROUGE-2 (F1) | ROUGE-L (Recall) | ROUGE-L (Precision) | ROUGE-L (F1) |
|------------------|------------------|---------------------|--------------|------------------|---------------------|--------------|------------------|---------------------|--------------|
| GPT-2            | 0.2989           | 0.1570              | 0.1976       | 0.0819           | 0.0440              | 0.0546       | 0.2523           | 0.1311              | 0.1657       |
| BART Base        | 0.3738           | 0.2444              | 0.3665       | -                | -                   | -            | -                | -                   | -            |
| BART Large-CNN   | 0.3257           | 0.2800              | 0.3257       | -                | -                   | -            | -                | -                   | -            |
| T5 Small         | 0.1835           | 0.0866              | 0.1732       | -                | -                   | -            | -                | -                   | -            |
| T5 Base          | 0.0985           | 0.0353              | 0.0959       | -                | -                   | -            | -                | -                   | -            |



![alt text](rouge_scores.png)

#### ROUGE Scores Explanation:

- **ROUGE-1** measures the overlap of unigrams (single words) between the generated and reference summaries. Higher scores indicate better performance in capturing essential words.

- **ROUGE-2** evaluates the overlap of bigrams (two consecutive words). It provides insight into how well the model captures pairs of words.

- **ROUGE-L** assesses the longest common subsequence between the generated and reference summaries. It reflects the fluency and coherence of the generated text.

### Model Information:

#### GPT-2:
1. **Architecture**: Generative Pre-trained Transformer 2.
2. **Training**: Trained on a diverse dataset for generating text.
3. **Strengths**: Good for creative and diverse text generation.
4. **Evaluation**: Lower ROUGE scores compared to BART models, but useful for generating varied outputs.
5. **Reference**: [GPT-2 on Hugging Face](https://huggingface.co/openai-community/gpt2)

#### BART Base:
1. **Architecture**: Bidirectional and Auto-Regressive Transformers.
2. **Training**: Pretrained on a large corpus of text and fine-tuned for summarization tasks.
3. **Usage**: Effective for generating coherent and fluent summaries.
4. **Evaluation**: High ROUGE scores, indicating good performance in generating relevant subject lines.
5. **Reference**: [BART Base on Hugging Face](https://huggingface.co/facebook/bart-base)

#### BART Large-CNN:
1. **Architecture**: Larger variant of BART with more parameters.
2. **Training**: Pretrained and fine-tuned for text summarization.
3. **Performance**: Balances recall and precision well, with high ROUGE scores.
4. **Usage**: Suitable for generating detailed and accurate summaries.
5. **Reference**: [BART Large-CNN on Hugging Face](https://huggingface.co/facebook/bart-large-cnn)

#### T5 Small:
1. **Architecture**: Text-To-Text Transfer Transformer.
2. **Training**: Converts all tasks to a text-to-text format.
3. **Efficiency**: Smaller model, faster inference.
4. **Performance**: Moderate ROUGE scores, balancing speed and accuracy.
5. **Reference**: [T5 Small on Hugging Face](https://huggingface.co/google-t5/t5-small)

#### T5 Base:
1. **Architecture**: Larger than T5 Small, with more parameters.
2. **Training**: Pretrained on a vast corpus and fine-tuned for specific tasks.
3. **Performance**: Provides a good balance of speed and accuracy.
4. **Usage**: Suitable for various text generation tasks with reasonable efficiency.
5. **Reference**: [T5 Base on Hugging Face](https://huggingface.co/google-t5/t5-base)


**Output for few Emails**:

1. Email : The following reports have been waiting for your approval for more than 4 days.Please review.Owner: James W Reitmeyer Report Name: JReitmeyer 10/24/01 Days In Mgr.Queue: 5
   Generated Subject: Following report waiting approval

