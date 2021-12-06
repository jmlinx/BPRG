import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

KWARGS = {'prog': 'twopi', 'root': 0}
NODE_SIZE = 23
FIGSIZE = (6, 6)

class PercolationViz():
    def __init__(self, A, kv_list, **kwargs):
        if not kwargs: kwargs = KWARGS
        self.A = A
        self.nv = A.shape[0]
        # threshold vectors through iterations
        self.kv_list = kv_list
        # get K
        K = len(np.unique(kv_list))-1
        self.K = K
        # generate edge list
        edges = self.adjacency_to_edges()
        self.edges = edges
        # index by group through iterations
        self.ikv_list = [
            self.index_by_group(kv, range(K+1))
            for kv in kv_list]
        # edge type by group through interations
        self.iedge_list = [
            self.check_infectious_edge(edges, ikv)
            for ikv in self.ikv_list]
        # generate networkx 
        G = nx.Graph()
        G.add_edges_from(self.edges)
        self.G = G
        self.pos = nx.nx_agraph.graphviz_layout(
            self.G, **kwargs)
        
        
        init_ikv = {0: [], 1: [], 2: list(range(self.nv))}
        init_iedge = {0: [], 1: [], 2: self.edges}
        self.ikv_list = [init_ikv] + self.ikv_list
        self.iedge_list = [init_iedge] + self.iedge_list
    
    def adjacency_to_edges(self):
        edges = [(i,j) for i in range(self.nv)
                 for j in range(self.nv)
                 if self.A[i,j]==1 and i < j]
        return edges

    def index_by_group(self, kv, k_list):
        return {k: np.where(kv==k)[0].tolist()
                    for k in k_list}
    
    
    def check_infectious_edge(self, edges, ikv):
        edge_dict = {0: [], 1:[], 2:[]}
        for edge in edges:
            if (edge[0] in ikv[0]) and (edge[1] in ikv[0]):
                edge_dict[0].append(edge)
            elif (edge[0] in ikv[0]) and (edge[1] not in ikv[0]):
                edge_dict[1].append(edge)
            else:
                edge_dict[2].append(edge)
        return edge_dict


    def plot_network_infection(self, G, pos, ikv,
                              iedge, figsize=FIGSIZE,
                              node_size=NODE_SIZE, ax=None):
        options = {
        'node_size': node_size,
        'alpha':0.9,    
        }
        node_color_list = ['violet', 'turquoise', 'springgreen']
        edge_color_list = ['violet', 'violet', 'springgreen']
        edge_options = {
            'width': 0.2,
            'alpha': 0.6,
        }

        for k in range(self.K + 1):
            nodelist = [i for i in ikv[k] if i in pos.keys()]
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist,
                                   node_color=node_color_list[k],
                                   **options)

            edgelist = iedge[k]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edgelist,
                                  edge_color=edge_color_list[k],
                                  **edge_options)
        plt.tight_layout()
        plt.axis("off")
    
    def get_percolation_plots(self, figsize=FIGSIZE, node_size=NODE_SIZE):
        G = self.G
        pos = self.pos
        ikv_list = self.ikv_list
        iedge_list = self.iedge_list

        figs = []
        
        for i in range(len(self.kv_list)):
            ikv = ikv_list[i]
            iedge = iedge_list[i]
            fig = plt.figure(figsize=figsize)
            self.plot_network_infection(
                G, pos, ikv, iedge, figsize, node_size, ax=None)
            figs.append(fig)            
        return figs
    
    def get_percolation_plot(self, nrows, ncols, idx_list,
                             figsize=(12,8), subfigsize=FIGSIZE,
                             node_size=NODE_SIZE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for i, ax in zip(idx_list, np.ravel(axes)):
            self.plot_network_infection(
                self.G, self.pos, self.ikv_list[i], self.iedge_list[i],
                node_size=node_size, figsize=subfigsize, ax=ax)
            ax.axis("off")
        return fig