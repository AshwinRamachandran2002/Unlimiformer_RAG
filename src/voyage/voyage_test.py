import voyageai 
import numpy as np
from voyageai import get_embeddings, get_embedding
from sklearn.metrics.pairwise import cosine_similarity


with open("combined_facts.txt", "r") as f:
    facts = f.read()
with open("persons.txt", "r") as f:
    person_list = f.read().splitlines()
with open("actions.txt", "r") as f:
    action_list = f.read().splitlines()

num_documents = 8
documents = facts.split(', ')[:num_documents]

voyageai.api_key = "tr_1a85945b5e1da8a900053c2e8ac2b4bed89eb88e5be0a5ba550a207cea85de4f094ab810b5fbdc4f1a5ec94debfd517ea1b31f3488b5f9517e6949d4c2823e1a"
documents_embeddings = get_embeddings(documents, model="voyage-01")


def k_nearest_neighbors(query_embedding, documents_embeddings, k=5):
  query_embedding = np.array(query_embedding) # convert to numpy array
  documents_embeddings = np.array(documents_embeddings) # convert to numpy array

  query_embedding = query_embedding.reshape(1, -1)

  cosine_sim = cosine_similarity(query_embedding, documents_embeddings)

  sorted_indices = np.argsort(cosine_sim[0])[::-1]

  top_k_related_indices = sorted_indices[:k]
  top_k_related_embeddings = documents_embeddings[sorted_indices[:k]]
  top_k_related_embeddings = [list(row[:]) for row in top_k_related_embeddings] # convert to list

  return top_k_related_embeddings, top_k_related_indices

for i in range(2 * num_documents):
    if i%2:
        query = 'Based on the above numbered list of facts, can you tell me the corresponding value for key: "' + person_list[i//2] + '"?'
    else:
        query = 'Based on the above numbered list of facts, can you tell me the corresponding key for value: "' + action_list[i//2] + '"?'

    query_embedding = get_embedding(query, model="voyage-01")

    retrieved_embd, retrieved_embd_index = k_nearest_neighbors(query_embedding, documents_embeddings, k=1)
    retrieved_doc = [documents[index] for index in retrieved_embd_index]

    if retrieved_doc[0] != documents[i//2]:
        print(i//2, "document failed")