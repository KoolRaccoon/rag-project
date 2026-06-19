*This is a simple RAG project*


Used the documentation from AWS lambda API reference guide https://docs.aws.amazon.com/lambda/latest/api/Welcome.html as data to read from. All files are converted to markdowns for ingestions

*The process of the RAG*

Load the data from /data folder
Split the data into chunks of 300 characters.
Creating embbedings for each chunk that convert the text into vectors of numbers using HuggingFace's all-MiniLM-L6-v2 model.

Save the text and the embbedings to a Chroma database, which will be used to search by similarity.

When a query is created it is embedded with the same model as the data and finds the 3 most similar chunks.

To build the response we are using groq llama-3.3-70b-versatile.
The query response is built out of 3 chunks as context into the ai model to generate the final answer.


The app uses streamlit as a minimal gui:

To run go to the main directory where main.py is located and run the following command:

`streamlit run main.py`

make sure you create a virtual env, install the requirements.txt and the start your venv before you run 

Here's an example of a query after the program is started:

![App Screenshot](assets/screenshot.png)
