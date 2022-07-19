import torch


def pairwise_distances(embeddings: torch.float32, metric: str = "euclidean") -> torch.float32:
    """Calculates pairwise distances of embeddings

    :param embeddings: embeddings
    :param metric: 'euclidean', 'euclidean_squared' or 'cosine'
    :return: pairwise distance matrix
    """

    if metric=="cosine":
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings_normalized = embeddings.div(norms.expand_as(embeddings))
        dists = torch.matmul(embeddings_normalized, embeddings_normalized.T)
        return 1.-dists

    # With help of: ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    dot_product = torch.matmul(embeddings, embeddings.T)
    square_norm = torch.diagonal(dot_product)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)
    distances = torch.maximum(torch.tensor(distances), torch.tensor(0.0))

    if metric=="euclidean_squared":
        return distances

    # Prevent square root from error with 0.0 -> sqrt(0.0)
    mask = torch.eq(distances, torch.tensor(0.0)).float()
    distances = distances + mask * 1e-16
    distances = torch.sqrt(distances)
    distances = distances * (1.0 - mask)
    return distances


def triplet_loss(distances: torch.float, mask_pos: torch.bool, mask_neg: torch.bool, margin: float) -> torch.float:
    """Triplet Loss Function

    :param distances: pairwise distances of all embeddings within batch
    :param mask_pos: mask of distance between A and P (positive distances)
    :param mask_neg: mask of distances between A and N (negative distances
    :param margin: the margin for the triplet loss
        Formula: Loss = max(0, dist(A,P) - dist(A,N) + margin)
    :return: triplet loss values
    """

    pos_dists = torch.multiply(distances, mask_pos)
    hardest_pos_dists = torch.amax(pos_dists, dim=1)
    neg_dists = torch.multiply(distances, mask_neg)
    neg_dists_max = torch.amax(neg_dists, dim=1, keepdim=True)
    dists_manipulated = distances + neg_dists_max * (1.0 - mask_neg)
    hardest_neg_dist = torch.amin(dists_manipulated, dim=1)

    return torch.maximum(hardest_pos_dists - hardest_neg_dist + margin, torch.tensor(0.0))


class OctupletLoss(torch.nn.modules.loss._Loss):
    def __init__(self, margin: float = 0.5, metric: str = "euclidean", configuration: list = None):
        """Octuplet Loss Function Generator
        See our paper -> TODO
        https://arxiv.TBD/
        See also ->
        https://omoindrot.github.io/triplet-loss (A nice Blog)

        :param margin: margin for triplet loss
        :param metric: 'euclidean', 'euclidean_squared', or 'cosine'
        :param configuration: configuration of triplet loss functions 'True' takes that specific loss term into account:
            Explanation: [Thhh, Thll, Tlhh, Tlll] (see our paper)
        :return: the octuplet loss function
        """

        super(OctupletLoss, self).__init__()
        if configuration is None:
            configuration = [True, True, True, True]
        self.margin = margin
        self.metric = metric
        self.configuration = configuration


    def forward(self, embeddings: torch.float, labels: torch.float) -> torch.float:
        """Octuplet Loss Function

        :param embeddings: concatenated high-resolution and low-resolution embeddings (size: 2*batch_size)
        :param labels: classes (size: batch_size)
        :return: loss value
        """

        # Concat embeddings with HR and LR images
        batch_size = labels.shape[0]
        pairwise_dist = pairwise_distances(embeddings, metric=self.metric)

        lbls_same = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
        not_eye_bool = torch.logical_not(torch.eye(batch_size, batch_size, device=lbls_same.device).bool())
        mask_pos = torch.eq(lbls_same, not_eye_bool).float()
        mask_neg = torch.logical_not(lbls_same).float()

        # TRIPLETS HR:HR ---------------------------------------------------------------
        dist_hrhr = pairwise_dist[0:batch_size, 0:batch_size]
        loss_hrhr = triplet_loss(dist_hrhr, mask_pos, mask_neg, self.margin)

        # TRIPLETS HR:LR ---------------------------------------------------------------
        dist_hrlr = pairwise_dist[0:batch_size, batch_size:2*batch_size]
        loss_hrlr = triplet_loss(dist_hrlr, mask_pos, mask_neg, self.margin)

        # TRIPLETS LR:HR ---------------------------------------------------------------
        dist_lrhr = pairwise_dist[batch_size:2*batch_size, 0:batch_size]
        loss_lrhr = triplet_loss(dist_lrhr, mask_pos, mask_neg, self.margin)

        # TRIPLETS LR:LR ---------------------------------------------------------------
        dist_lrlr = pairwise_dist[batch_size:2*batch_size, batch_size:2*batch_size]
        loss_lrlr = triplet_loss(dist_lrlr, mask_pos, mask_neg, self.margin)

        # Combination of triplet loss terms
        losses = torch.transpose(torch.tensor([self.configuration] * batch_size), 0, 1) * torch.stack([
            loss_hrhr,
            loss_hrlr,
            loss_lrhr,
            loss_lrlr,
        ])
        summe = torch.sum(losses, dim=0)
        total_loss = torch.mean(summe, dim=0)

        return total_loss
