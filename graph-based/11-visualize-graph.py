import os
import networkx as nx
import matplotlib.pyplot as plt
from conllu import parse_incr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def read_conllu_file(conllu_path):
    """Read a CoNLL-U file and return the parsed sentence."""
    try:
        if not os.path.exists(conllu_path):
            raise FileNotFoundError(f"CoNLL-U file not found: {conllu_path}")
        
        with open(conllu_path, 'r', encoding='utf-8') as f:
            sentences = list(parse_incr(f))
        
        if not sentences:
            raise ValueError(f"No sentences found in {conllu_path}")
        
        logger.info(f"Parsed CoNLL-U file: {conllu_path}")
        return sentences[0]  # Return first sentence (each file has one)
    
    except Exception as e:
        logger.error(f"Error reading CoNLL-U file: {str(e)}")
        raise

def build_dependency_graph(sentence):
    """Build a directed graph from a CoNLL-U sentence."""
    G = nx.DiGraph()
    
    # Add nodes (words) and edges (dependencies)
    for token in sentence:
        token_id = token['id']
        word = token['form']
        pos = token['upos']
        head = token['head']
        deprel = token['deprel']
        
        # Node label: word/POS
        G.add_node(token_id, label=f"{word}/{pos}")
        
        # Add edge from head to dependent (if head exists and is not root)
        if head is not None and head != 0:
            G.add_edge(head, token_id, label=deprel)
    
    return G

def visualize_graph(G, sent_id, output_dir):
    """Visualize the dependency graph and save it."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"dependency_graph_{sent_id}.png")
        
        # Hierarchical layout for tree structure
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        plt.figure(figsize=(12, 8))
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20)
        # Draw node labels (word/POS)
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        # Draw edge labels (dependency relations)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Dependency Graph for Sentence ID: {sent_id}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300)
        logger.info(f"Saved graph to {output_path}")
        plt.show()
        plt.close()
    
    except Exception as e:
        logger.error(f"Error visualizing graph: {str(e)}")
        raise

def main():
    # Paths
    base_path = "D:\\Manish Prajapati\\AML Sounding Trees\\Project\\Growing_tree_on_sound\\graph_based"
    conllu_dir = os.path.join(base_path, "conllu")
    output_dir = os.path.join(base_path, "graphs")
    
    # Specify sentence ID to visualize (using one from your list)
    sent_id = "6308-68359-0026"  # Replace with any of: 454-134728-0058, 548-126410-0002, 698-123197-0022, 949-134660-0000
    conllu_path = os.path.join(conllu_dir, f"valid_{sent_id}.conllu")
    
    # Read and visualize
    sentence = read_conllu_file(conllu_path)
    graph = build_dependency_graph(sentence)
    visualize_graph(graph, sent_id, output_dir)

if __name__ == "__main__":
    main()