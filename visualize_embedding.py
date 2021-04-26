import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gensim

def visualize_embedding(model):
    nodes = [x for x in model.wv.vocab]
    embedding = np.array([model.wv[x] for x in nodes])
    labels = [1 if int(node)<=871 else 0 for node in nodes]
    writer = SummaryWriter('runs/visualize_embeddings')
    writer.add_embedding(embedding, metadata=labels)
    writer.close()

def create_tsv(model):
    nodes = [x for x in model.wv.vocab]
    embedding = np.array([model.wv[x] for x in nodes])
    labels = [1 if int(node)<=871 else 0 for node in nodes]

    with open("./tensor.tsv","a") as f:
        f.truncate(0)
        for e in embedding:
            e = list(map(lambda x:str(x),e))
            f.write("\t".join(e) + "\n")
        f.close()
    with open("./metadata.tsv","a") as f:
        f.truncate(0)
        f.write("Id\tLabel\n")
        for i in range(len(nodes)):
            f.write(f"{nodes[i]}\t{labels[i]}\n")
        f.close()

if __name__ == "__main__":
    model = gensim.models.Word2Vec.load("./model_5.bin")
    # visualize_embedding(model)
    create_tsv(model)