from scipy.cluster.hierarchy import linkage
from collections import deque
import torch
        
class TreeNode:
    def __init__(self, idx, left=None, right=None, distance=0.0, size=1):
        self.idx = int(idx)
        self.left: TreeNode = left
        self.right: TreeNode = right
        self.distance = distance
        self.size = size # Number of Node in this subtree
        
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def iter(self):
        results = []
        stack = deque()
        last_visited = None
        node = self
        
        while node or stack:
            if node:
                stack.append(node)
                node = node.left
            else:
                top_node = stack[-1]
                if top_node.right and last_visited != top_node.right:
                    node = top_node.right
                else:
                    if top_node.is_leaf():
                        results.append(top_node.idx)
                    last_visited = stack.pop()
        return results
    
    def __str__(self):
        return str(self.idx)
    def __repr__(self):
        return str(self.idx)
        
        
class BinaryClusterTree:
    def __init__(self, min_clu_size=3):
        self.min_clu_size = min_clu_size
        self.root = None
        self.adj = None
        
    def fit(self, x):
        n_samples, _ = x.shape
        x = torch.where(torch.isfinite(x), x, 1e10)
        linkage_matrix = linkage(x, method='average')
        self.root = self._build_tree(linkage_matrix, n_samples)
        return self.root
        
    def classify(self):
        return self._remove_outliers(self.root, self.min_clu_size)
        
    @staticmethod
    def _build_tree(linkage_mat, n_samples):
        nodes = {}
        cur_idx = n_samples
        for idx1, idx2, dist, size in linkage_mat:
            if idx1 < n_samples:
                left = TreeNode(idx=idx1, size=1)
            else:
                left = nodes.pop(int(idx1))
            
            if idx2 < n_samples:
                right = TreeNode(idx=idx2, size=1)
            else:
                right = nodes.pop(int(idx2))
                
            
            root = TreeNode(idx=int(cur_idx), left=left, right=right, distance=dist, size=size)
            nodes[cur_idx] = root
            cur_idx = cur_idx + 1
        
        return root
    
    @staticmethod
    def _remove_outliers(root: TreeNode, min_clu_size):
        outliers = []
        n_clients = root.size
        
        while root is not None:
            left_size = root.left.size if root.left else 0
            right_size = root.right.size if root.right else 0
            
            if left_size <= min_clu_size or right_size <= min_clu_size:
                if root.right and root.right.size >= min_clu_size:
                    outlier = root.left.iter() if root.left else []
                    root = root.right
                elif root.left and root.left.size >= min_clu_size:
                    outlier = root.right.iter() if root.right else []
                    root = root.left
                else:
                    outlier = root.iter()
                    outliers.extend(outlier)
                    root = None
                    break
                outliers.extend(outlier)
                
                if len(outliers) > (n_clients // 2):
                    break
            else:
                break
            
        benign, malicious = [], []
        if not root:
            return benign, malicious, outliers
        
        left_size = root.left.size if root.left else 0
        right_size = root.right.size if root.left else 0
        thr = n_clients // 2
        
        if left_size > thr or right_size > thr:
            benign_subtree = root.left if () else root.right
            malicious_subtree = root.right if left_size > right_size else root.left
            benign = benign_subtree.iter()
            malicious = malicious_subtree.iter()
        else:
            benign = root.iter()
        
        return benign, malicious, outliers