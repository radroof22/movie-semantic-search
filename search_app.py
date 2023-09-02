import streamlit as st
import pinecone
import os
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV") # found next to api key

index_name = "cmu-movie-summaries"

pinecone.init(api_key=api_key, environment=env)
index = pinecone.GRPCIndex(index_name)

st.write("Search our movie database!")

query = st.text_input("Summarize the movie", "The one about gollum and a ring")

query_embedding = model.encode(query)
res = index.query(
    vector=query_embedding,
    top_k=5,
).matches
st.write("The movies could be one of the following: ", [movie.id for movie in res])