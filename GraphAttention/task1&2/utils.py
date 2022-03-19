import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import sys
import pickle as pkl

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_citeseer(dataset_str='citeseer'):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print(adj.shape, np.sum(adj), np.sum(adj.todense()[0]))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print(adj.shape, np.sum(adj), np.sum(adj.todense()[0]))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    print(adj.shape, np.sum(adj), np.sum(adj.todense()[0] ), np.sum(adj.todense()[0] > 0), np.where(adj.todense()[0] > 0))

    features = normalize_features(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))
    #idx_train = range(len(y))
    #idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    nw_adj_2 = get_more_oder(adj)
    adj = [np.array(adj.todense()), nw_adj_2]

    adj = torch.FloatTensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    print(labels)

    train_mask = torch.LongTensor(train_mask)
    val_mask = torch.LongTensor(val_mask)
    test_mask = torch.LongTensor(test_mask)

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj, features, labels, train_mask, val_mask, test_mask
    # return adj, features, labels, idx_train, idx_val, idx_test


def get_more_oder(adj):
    adj_np = np.array(adj) > 0
    nw_adj_2 = np.zeros_like(adj_np) + 0.0
    # print("##", np.shape(nw_adj_2), np.amax(nw_adj_2), np.amin(nw_adj_2))
    print(np.where(adj_np[344] == 1), np.where(adj_np[170] == 1),np.where(adj_np[258] == 1))

    Lnodes = {}
    for i in range(np.shape(adj_np)[0]):
        loc = np.where(adj_np[i] == 1)[0]
        Lnodes[i] = loc
        if i < 10:
            print(i, Lnodes[i], np.sum(adj_np[i]))

    # print("########")
    for i in range(np.shape(adj_np)[0]):
        for j in Lnodes[i]:
            for z in Lnodes[j]:
                if z in Lnodes[i]:
                    nw_adj_2[i, z] = 1.0
                else:
                    nw_adj_2[i, z] = 1.0
            # if i < 10:
            #     print(i, j, Lnodes[j])
    # print("##", np.shape(nw_adj_2), np.amax(nw_adj_2), np.amin(nw_adj_2), np.sum(nw_adj_2[0]), np.sum(nw_adj_2[1]), np.sum(nw_adj_2[2]), np.sum(nw_adj_2[3]))
    print("neighbor-2")
    for i in range(np.shape(nw_adj_2)[0]):
        loc = np.where(nw_adj_2[i] == 1)[0]
        if i < 10:
            print(i, loc)#, end=' - ')
        # loc = np.where(nw_adj_2[i] == 2.0)[0]
        # if i < 10:
        #     print(loc)

    return nw_adj_2

# def load_data(path="./data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))

#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])

#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     features = normalize_features(features)
#     adj = normalize_adj(adj + sp.eye(adj.shape[0]))

#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)

#     nw_adj_2 = get_more_oder(adj)
#     adj = [np.array(adj.todense()), nw_adj_2]
#     adj = torch.FloatTensor(adj)
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])

#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)

#     return adj, features, labels, idx_train, idx_val, idx_test


# def load_data(path="./data/cora/", dataset="cora"):
def load_data(path="./data/citeseer/", dataset="citeseer", order=2):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = idx_features_labels[:, 0]#np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx = np.arange(0, idx) + 999000
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))#, dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # edges = list(map(idx_map.get, edges_unordered.flatten()))


    edges_tmp = [(x, y) for (x, y) in edges if x is not None and y is not None ]
    edges_tmp = np.array(edges_tmp)


    adj = sp.coo_matrix((np.ones(edges_tmp.shape[0]), (edges_tmp[:, 0], edges_tmp[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    nw_adj_2 = get_more_oder(adj.todense())
    if order == 3:
        nw_adj_3 = get_more_oder(nw_adj_2)

    if dataset == 'cora':
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset == 'citeseer':
        idx_train = range(120)
        idx_val = range(1200, 1700)
        idx_test = range(200, 1200)

    print(np.amax(labels))

    if order == 3:
        adj = [np.array(adj.todense()), nw_adj_2, nw_adj_3]
    else:
        adj = [np.array(adj.todense()), nw_adj_2]
    # adj = [(np.array(adj.todense())>0)+0.0, nw_adj_2]
    adj = torch.FloatTensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    print(torch.amax(labels), torch.amin(labels), torch.unique(labels))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

