import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Attention-based Agreggation Layer: 
            From a M : n x h tiles features matrix, computes attention scores a_1, ..., a_n for 
            each line (each tile) L_i of M then returns the weighted mean of the n features lines :
                    Sum_i a_i L_i
            
            Attention scores are defined as follows (*):
                    l_i = W.(tanh(U . L_i) * sigmoid(V . L_i)) 
                        where . is the matrix product and * the hadamard product (element-wise)

                    a_i = exp(l_i) / [Sum_j exp(l_j)] = softmax(l)_i
            
            Gated attention mechanism defined in https://arxiv.org/pdf/1802.04712.pdf
    """
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.V = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.W = torch.nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, M):
        u = self.U(M)
        u = torch.tanh(u)
        
        v = self.V(M)
        v = torch.sigmoid(v)
        
        w = u*v
        w = self.W(w)
        
        a = F.softmax(w, dim=-2)
        
        out = a * M
        out = out.sum(dim=-2)
        
        return a, out

class DSAttentionLayer(nn.Module):
    """Attention-based Aggregation Layer: 
            From a M : n x h tiles features matrix, computes attention scores a_1, ..., a_n for 
            each line (each tile) L_i of M then returns the weighted mean of the n features lines :
                    Sum_i a_i L_i

            Attention scores are defined as follows (*):
                L_m = argmax(L_0.W0, ..., L_n.W_0)
                
                Vi, q_i = L_i.Q and v_i = L_i.V
                Learnable distance between L_i and the argmax L_m:
                    U(L_i, L_m) = exp(<q_i, q_m>) / Sum_k exp(<q_k, q_m>)
                
                Then all tiles representations can be aggregated:
                    b = Sum_i U(L_i, L_m) v_i
                
            Defined in https://arxiv.org/pdf/2011.08939.pdf
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W0 = torch.nn.Linear(input_dim, 1, bias=False)
        self.Q = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.V = torch.nn.Linear(input_dim, input_dim, bias=False)
    
    def forward(self, M):
        scores = self.W0(M)
        scores = torch.sigmoid(scores)
        id_max = torch.argmax(scores)
        score_m = scores[0, id_max]

        q = self.Q(M)
        v = self.V(M)

        q_m = q[0, id_max]

        #a : 1 x N x 1
        a = torch.matmul(q, q_m)
        a = F.softmax(a, -1)
        a = a.unsqueeze(2)

        out = a * v 
        out = out.sum(dim=-2)

        return a, out, score_m
    
class MixedMILModel(nn.Module):
    """Mixed MIL Model inspired by https://proceedings.mlr.press/v156/tourniaire21a/tourniaire21a.pdf
           Input shape ((n, input_dim)): n tiles features of dim input_dim.
           1st layer: input_dim -> dim1
                Slide classifier
                   2nd layer (input shape (n, dim1)): aggregate the tiles features thanks to an attention layer. 
                   3rd layer (input shape (dim1,)): dim1 -> dim2
                   4th layer (input shape (dim2,)): binary classification dim2 -> 1 
                Tiles classifier
                   2nd layer (input shape (n, dim1)) : binary classification dim1 -> 1
            
            NB: In the training procedure, slide classifier attention scores are used to produced tiles 
            pseudo-labels to compare with the tiles classifier outputs.
    """ 
    def __init__(self, input_dim, dim1, dim2, attention_dim, ds=False, p=0.25):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.ds = ds

        self.reduction = nn.Sequential(nn.Dropout(p),
                                       nn.Linear(input_dim, dim1),
                                       nn.ReLU())
        
        self.attention_dim = attention_dim

        if ds:
            self.attention = DSAttentionLayer(dim1, attention_dim)
        else:
            self.attention = AttentionLayer(dim1, attention_dim)
        
        self.slide_clf = nn.Sequential(nn.Dropout(p),
                                 nn.Linear(dim1, dim2),
                                 nn.ReLU(),
                                 nn.Dropout(p),
                                 nn.Linear(dim2, 1),
                                 nn.Sigmoid())
        
        self.tiles_clf = nn.Sequential(nn.Dropout(p),
                                       nn.Linear(dim1, 1),
                                       nn.Sigmoid())
        
    def forward(self, M):
        M = self.reduction(M)
        if not self.ds:
            a, r = self.attention(M)
            slide_out = self.slide_clf(r)[:,0]
        else:
            a, r, s = self.attention(M)
            slide_out = (s + self.slide_clf(r)[:,0]) / 2
        
        tiles_out = self.tiles_clf(M)

        return slide_out, tiles_out, a[:,:,0]
    
    def reset_parameters(self):
        for layer in self.children():
            for c in layer.children():
                if hasattr(c, 'reset_parameters'):
                    c.reset_parameters()