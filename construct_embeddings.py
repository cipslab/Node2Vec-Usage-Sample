import networkx as nx
import numpy as np
import argparse
import logging
from node2vec import Node2Vec

logging.basicConfig(level=logging.DEBUG)

def build_graph(edge_path, allow_self_loop=False):
    with open(edge_path,"r") as f:
        edges = f.readlines()
    G = nx.DiGraph()
    for e in edges:
        e = list(map(lambda x:int(x),e.strip().strip("\n").split("\t")))
        G.add_edge(e[0],e[1])
    if not allow_self_loop:
        G.remove_edges_from(nx.selfloop_edges(G))
    return G 


def main():
    logging.debug(f"Graph being built from {args.edges_path}\n")
    G = build_graph(args.edges_path)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    if args.k_cores:
        logging.debug(f"K Cores starting with cores={args.cores}\n")
        G = nx.k_core(G, args.cores)
        logging.debug(f"K Cores Finished\n")
        print(f"Pruned Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    logging.debug(f"""Node2Vec Starting\n\nHyperparameters:\ndimensions={args.d}\n
                    walk_len={args.walk_len},
                    num_walks={args.num_walks},
                    workers={args.workers},
                    p={args.p},
                    q={args.q}""")

    node2vec = Node2Vec(G,
                    dimensions=args.d,
                    walk_length=args.walk_len,
                    num_walks=args.num_walks,
                    workers=args.workers,
                    p=args.p,
                    q=args.q,
                    temp_folder="/mnt/data/")
    
    model = node2vec.fit(window=args.context_size)
    logging.debug(f"Model saving to {args.O}")
    model.save(args.O)
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--edges_path",
                        default="./ahrrefs-edges.txt",
                        help="data path")
    parser.add_argument("--k_cores",
                        action="store_false",
                        help="Apply K cores or not")
    parser.add_argument("--cores",
                        default=5,
                        type=int,
                        help="Number of cores to prune")
    parser.add_argument("--p",
                        default=1,
                        type=float,
                        help="return parameter")
    parser.add_argument("--q",
                        default=1,
                        type=float,
                        help="skip parameter")
    parser.add_argument("--walk_len",
                        default=80,
                        type=int,
                        help="Random walk length")
    parser.add_argument("--num_walks",
                        default=100,
                        type=int,
                        help="Number of random walks per node")
    parser.add_argument("--d",
                        default=64,
                        help="Dimensionality of node representation")
    parser.add_argument("--context_size",
                        default=10,
                        help="Context size to consider for skipgram")
    parser.add_argument("--workers",
                        default=10,
                        help="Number of workers to use for node2vec")
    parser.add_argument("--O",
                        default="model.bin",
                        help="output path for node2vec output")
    
    args = parser.parse_args()

    main()