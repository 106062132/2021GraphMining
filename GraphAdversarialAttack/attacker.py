import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
import utils
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import scipy.sparse as sp


class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device

        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

        self.modified_adj = None
        self.modified_features = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        pass

    def check_adj(self, adj):
        """Check if the modified adjacency is symmetric and unweighted.
        """

        if type(adj) is torch.Tensor:
            adj = adj.cpu().numpy()
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        if sp.issparse(adj):
            assert adj.tocsr().max() == 1, "Max value should be 1!"
            assert adj.tocsr().min() == 0, "Min value should be 0!"
        else:
            assert adj.max() == 1, "Max value should be 1!"
            assert adj.min() == 0, "Min value should be 0!"

    def save_adj(self, root=r'/tmp/', name='mod_adj'):
        """Save attacked adjacency matrix.
        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name
        Returns
        -------
        None.
        """
        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj

        if type(modified_adj) is torch.Tensor:
            sparse_adj = utils.to_scipy(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)
        else:
            sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):
        """Save attacked node feature matrix.
        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name
        Returns
        -------
        None.
        """

        assert self.modified_features is not None, \
                'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            sparse_features = utils.to_scipy(modified_features)
            sp.save_npz(osp.join(root, name), sparse_features)
        else:
            sp.save_npz(osp.join(root, name), modified_features)


class RND(BaseAttack):
    """As is described in Adversarial Attacks on Neural Networks for Graph Data (KDD'19),
    'Rnd is an attack in which we modify the structure of the graph. Given our target node v,
    in each step we randomly sample nodes u whose label is different from v and
    add the edge u,v to the graph structure
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    Examples
    --------
    >>> from dataset import Dataset
    >>> from attacker import RND
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = RND()
    >>> # Attack
    >>> model.attack(adj, labels, idx_train, target_node, n_perturbations=5)
    >>> modified_adj = model.modified_adj
    >>> modified_features = model.modified_features
    """

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=True, device='cpu'):
        super(RND, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)

    def attack(self, surrogate, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, **kwargs):
        """
            Randomly sample nodes u whose label is different from v and
            add the edge u,v to the graph structure. This baseline only
            has access to true class labels in training set
            Parameters
            ----------
            ori_features : scipy.sparse.csr_matrix
            Origina (unperturbed) node feature matrix.
            ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
            labels :
            node labels
            idx_train :
            node training indices
            target_node : int
            target node index to be attacked
            n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
            """
        # ori_adj: sp.csr_matrix
        
        print('number of pertubations: %s' % n_perturbations)
        modified_adj = ori_adj.tolil()
        modified_features = ori_features.tolil()
        
        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        
        change_array = []
        for k in range(n_perturbations):
            max_error = 0
            max_index = 0
            for i in range(len(diff_label_nodes)):
                if(i not in change_array):
                    changed_nodes = diff_label_nodes[i]
                    modified_adj[target_node, changed_nodes] = 1
                    modified_adj[changed_nodes, target_node] = 1
                    adj_output = surrogate.predict(modified_features,modified_adj)
                    error,_ = utils.loss_acc(adj_output,labels,target_node)
                    if(error.item() > max_error):
                        max_index = i
                        max_error = error
                    modified_adj[target_node, changed_nodes] = 0
                    modified_adj[changed_nodes, target_node] = 0
            change_array.append(max_index)
            changed_nodes = diff_label_nodes[max_index]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        self.modified_features = modified_features
