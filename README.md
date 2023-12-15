# RAGAS 
Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. This repo is customized for my internship project.

**Thank to [RAGAS](https://docs.ragas.io/en/latest/) for this useful framework**

# Modifications
* Modifying prompts when calculating `faithfulness`, `context_recall` to ignore social interaction, email sentences.
* Adding a new metric `request_recall` to measure the extent to which the extracted requests align with the question.
* Logging to file
* Adding intermediate LLM's responses into output files for analysis.