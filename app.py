from datasets import load_dataset
import chromadb
from chromadb.config import Settings
import spacy
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import from_networkx
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn
import re
from chromadb.config import Settings
import traceback

sp_model= spacy.load("en_core_sci_md")
#print(sp_model.pipe_names)
model = SentenceTransformer('all-MiniLM-L6-v2')

QA_dataset= load_dataset("enelpol/rag-mini-bioasq","question-answer-passages")
Passages_dataset=load_dataset("enelpol/rag-mini-bioasq", "text-corpus")
#print(Passages_dataset["test"]["passage"][0]) #just checking a sample
print("passage structure:", Passages_dataset["test"].features.keys()) #passage with id
print("QA structure:",QA_dataset.keys())
print("qa test structure:", QA_dataset["test"].features.keys()) #passage with id
print("qa train structure:", QA_dataset["train"].features.keys()) #passage with id


def clean_text(text):
    text = re.sub(r'\s+', ' ', text) #keeping a single space bs
    text = text.strip() #shel spaces f awl w akhr gomla 
    return text

cleaned_data=[]
for split in Passages_dataset.keys():
  for txt in Passages_dataset[split]:
    data=txt.get("passage","")
    clean=clean_text(data)
    cleaned_data.append({
        "id":txt["id"],
        "passage":clean
            })

if cleaned_data:
    print("First cleaned:", cleaned_data[0])
else:
    print("No data was processed")


def build_graph(passage):
   data=sp_model(passage)
   entities=[]
   graph=nx.Graph()
   for ent in data.ents:
      entity={
         'text':ent.text,
         'type':ent.label_,
         'embedding': model.encode([ent.text])[0]   
      }
      entities.append(entity)
      graph.add_node(ent.text, **entity)
    
        
   for line in data.sents:
      sent_entities=[]
      for e in entities:
         if e['text'] in line.text:
            sent_entities.append(e)
         for i in range(len(sent_entities)):
            for j in range(i+1,len(sent_entities)):
               first=sent_entities[i]['text']
               sec=sent_entities[j]['text']
               graph.add_edge(first,sec,relation='occurrence')
   return graph
           
           
def convert(graph):
   node_embeddings=[]
   for n in graph.nodes(data=True):
      node_embeddings.append(n[1]["embedding"])
   node_tensor = torch.tensor(node_embeddings, dtype=torch.float)

   edges=[]
   edge_rel=[]
   for edge in graph.edges(data=True):
    first, second, data = edge
    edges.append([list(graph.nodes).index(first), list(graph.nodes).index(second)])
    edge_rel.append(1)  #f nfs scentence fe edge-->1

   edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
   edge_attr_tensor = torch.tensor(edge_rel, dtype=torch.float).view(-1, 1) 

   pyg_graph = from_networkx(graph)
   pyg_graph.x = node_tensor
   pyg_graph.edge_index = edge_index_tensor
   pyg_graph.edge_attr = edge_attr_tensor
   
   return pyg_graph


example_passage = cleaned_data[0]['passage']
graph = build_graph(example_passage)
pyg_graph = convert(graph)
print("PyG node features shape:", pyg_graph.x.shape)
print("PyG edge index shape:", pyg_graph.edge_index.shape)


class GraphTrans(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=384, heads=2):
        super(GraphTrans, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)
        self.linear = nn.Linear(output_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index    
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        # Pooling
        batch = torch.zeros(x.size(0), dtype=torch.long)
        graph_embedding = global_mean_pool(x, batch)
        
        # normal->cos similarity
        return torch.nn.functional.normalize(
            self.linear(graph_embedding), 
            p=2, dim=1
        ).squeeze(0)

transformer = GraphTrans()

def process_passage(passage):
    try:    
        graph = build_graph(passage['passage']) 
        pyg_graph = convert(graph)
        with torch.no_grad():
            embedding = transformer(pyg_graph).detach().numpy()  
            
        return {
            'id': str(passage['id']),
            'embedding': embedding,
            'document': passage['passage'][:500]
        }
    except Exception as e:
        print(f"Error processing passage {passage['id']}: {str(e)}")
        return None
    

    
chroma_client = chromadb.PersistentClient(
    path=".chromadb/",
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(name="passage_embeddings")


def store_passages(passages_data, batch_size=50):
    results = []
    for i, passage in enumerate(tqdm(passages_data, desc="Processing")):
        result = process_passage(passage)
        if result:
            results.append(result)
        if len(results) >= batch_size:
            try:
                print(f"Adding batch of {len(results)} items to collection")
                collection.add(
                    ids=[r['id'] for r in results],
                    documents=[r['document'] for r in results],
                    embeddings=[r['embedding'].tolist() for r in results]
                )
                results = []
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {str(e)}")
                results = [] 
    
    if results:
        try:
            print(f"Adding final batch of {len(results)} items")
            collection.add(
                ids=[r['id'] for r in results],
                documents=[r['document'] for r in results],
                embeddings=[r['embedding'].tolist() for r in results]
            )
            processed += len(results)
            print(f"Successfully added final batch. Total processed: {processed}")
        except Exception as e:
            print(f"Error adding final batch to ChromaDB: {str(e)}")

# Run processing
r = process_passage(cleaned_data[0])
#l error hna blzbt 3nd collection.add aw upsert, hwa by terminate alatol
collection.upsert(
    ids=[r['id']],
    documents=[r['document']],
    embeddings=[r['embedding'].tolist()]
)

store_passages(cleaned_data,batch_size=50)
