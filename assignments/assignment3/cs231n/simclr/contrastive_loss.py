import torch
import numpy as np

# 计算余弦相似度
def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # 计算 z_i 和 z_j 的点积，然后除以它们 L2 范数的乘积，得到余弦相似度
    norm_dot_product = torch.dot(z_i, z_j) / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # 计算 l(k, k+N)：分子是 z_k 和 z_k_N 的相似度指数
        # 分母是 z_k 与其他所有样本的相似度指数之和
        
        num_1 = torch.exp(sim(z_k, z_k_N) / tau)
        den_1 = sum(torch.exp(sim(z_k, out[t]) / tau) for t in range(2 * N) if t != k)
        loss_k_k_N = -torch.log(num_1 / den_1)

        # 计算 l(k+N, k)：分子是 z_k_N 和 z_k 的相似度指数
        # 分母是 z_k_N 与其他所有样本的相似度指数之和
        num_2 = torch.exp(sim(z_k_N, z_k) / tau)
        den_2 = sum(torch.exp(sim(z_k_N, out[t]) / tau) for t in range(2 * N) if t != k + N)
        loss_k_N_k = -torch.log(num_2 / den_2)

        # 将对称的两个损失累加
        total_loss += loss_k_k_N + loss_k_N_k

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # 分别计算 out_left 和 out_right 每一行的 L2 范数，保持维度以便广播
    # dim=1 表示在每行内部横向计算 L2 范数。
    norm_left = torch.linalg.norm(out_left, dim=1, keepdim=True)   # [N, 1]
    norm_right = torch.linalg.norm(out_right, dim=1, keepdim=True) # [N, 1]
    
    # 计算每一行对应样本的点积（矩阵逐元素相乘后dim=1对每行求和）
    # 然后逐元素除以范数乘积，得到 Nx1 的余弦相似度
    pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True) / (norm_left * norm_right)

    # 形状为 [N, 1] 的张量 pos_pairs，
    # 其中第 k 行就是 out_left[k] 和 out_right[k] 之间的归一化点积（余弦相似度）
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    # 计算 out 每一行的 L2 范数，保持维度为 [2N, 1]
    norms = torch.linalg.norm(out, dim=1, keepdim=True)
    
    # 通过矩阵乘法计算所有样本两两之间的点积，然后除以范数乘积得到余弦相似度矩阵
    # 分母加上 1e-8 ，防止除以零，保证数值稳定性
    sim_matrix = torch.mm(out, out.T) / (torch.mm(norms, norms.T) + 1e-8)

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
   
    exponential = torch.exp(sim_matrix / tau)

    # This binary mask zeros out terms where k=i.
    # 用全 1 矩阵减去单位矩阵，得到一个对角线为 0，其余为 1 的掩码矩阵
    # 这样在计算分母时就不会把自身的相似度算进去
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    # 利用掩码剔除自身与自身的相似度，并重塑为 [2*N, 2*N-1]
    # masked_select() 返回一个一维张量，包含 exponential (bool矩阵)中所有 mask 为 True 的元素
    # 然后用 view(,-1) 自动计算维度， 重塑为 [2*N, 2*N-1]
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    # 对每一行求和，得到每个样本的分母项（排除了自身），形状为 [2*N, 1]
    denom = torch.sum(exponential, dim=1, keepdim=True)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    
    # 提取正样本对的相似度，形状为 [N, 1]
    sim_pos = sim_positive_pairs(out_left, out_right)
    
    # Step 3: Compute the numerator value for all augmented samples.
    # 将正样本对的相似度拼接两次，形状变为 [2*N, 1]，以对应左分支和右分支的样本
    numerator = torch.exp(torch.cat([sim_pos, sim_pos], dim=0) / tau) # [2*N, 1]
    
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    # 计算每个样本的损失，然后求所有 2*N 个样本的平均值
    loss = -torch.log(numerator / denom).mean()
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))