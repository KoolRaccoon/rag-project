This is a simple RAG project 


Used the documentation from AWS lambda files as data to read from. All files are converted to markdowns for ingestions

The process of the RAG

Load the data 
Split the data into chunks of 300 characters.
Creating embbedings for each chunk that convert the text into vectors of numbers using HuggingFace's all-MiniLM-L6-v2 model.

Save the text and the embbedings to a Chroma database, which will be used to search by similarity.

When a query is created it is embedded with the same model as the data and find the 3 most similar chunks.

To build the response we are using groq llama-3.3-70b-versatile.
The query response is built out of 3 chunks as context into the ai model to generate the final answer.



