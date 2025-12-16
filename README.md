# Amazon Reviews
Using the amazon reviews 2023 database to design a recommendation system and q &amp; a system for users.

# Data
https://drive.google.com/drive/u/2/folders/1iJBBhTgFMMBKMTEKQvvzS8gRt6whLH4x  
The above link contains preprocessed embeddings for the **appliances** subset of the data found [here](https://amazon-reviews-2023.github.io/#load-user-reviews).  
These embeddings were generated using the embed.py file. Additional embeddings for other categories can also be generated using this method.  
When testing using this dataset, please keep in mind that it only contains data on appliances (fridges, washing machines, etc). 

# Usage
The entry point for this project is **semantic.py**. To run, simply execute **python semantic.py**.  
You will first be asked to input a set of requirements, which will be used to recommend an initial list of products.  
Afterwards, you can select products from this list and ask detailed questions about each of them. 
