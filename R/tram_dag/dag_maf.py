import numpy as np
#### Function with creates the masks for the MLP
def create_masks_np(adjacency, hidden_features=(64, 64), activation='relu'):
    out_features, in_features = adjacency.shape
    adjacency, inverse_indices = np.unique(adjacency, axis=0, return_inverse=True)
    precedence = np.dot(adjacency.astype(int), adjacency.T.astype(int)) == adjacency.sum(axis=-1, keepdims=True).T
    masks = []
    for i, features in enumerate((*hidden_features, out_features)):
        if i > 0:
            mask = precedence[:, indices]
        else:
            mask = adjacency
        if np.all(~mask):
            raise ValueError("The adjacency matrix leads to a null Jacobian.")

        if i < len(hidden_features):
            reachable = np.nonzero(mask.sum(axis=-1))[0]
            if len(reachable) > 0:
                indices = reachable[np.arange(features) % len(reachable)]
            else:
                indices = np.array([], dtype=int)
            mask = mask[indices]
        else:
            mask = mask[inverse_indices]
        masks.append(mask)
    return masks

adjacency = np.array([[0,0,0],[1,0,0],[1,0,0]]) == 1
hidden_features=[3,3]
create_masks_np(adjacency, hidden_features=hidden_features)
