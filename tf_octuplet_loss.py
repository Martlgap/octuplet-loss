import tensorflow as tf


@tf.function
def pairwise_distances(embeddings: tf.float32, metric: str = "euclidean") -> tf.float32:
    """Calculates pairwise distances of embeddings

    :param embeddings: embeddings
    :param metric: 'euclidean', 'euclidean_squared' or 'cosine'
    :return: pairwise distance matrix
    """

    if metric == "cosine":
        distances_normalized = tf.nn.l2_normalize(embeddings, axis=1)
        distances = tf.matmul(distances_normalized, distances_normalized, adjoint_b=True)
        return 1.0 - distances

    # With help of: ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if metric == "euclidean_squared":
        return distances

    # Prevent square root from error with 0.0 -> sqrt(0.0)
    mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
    distances = distances + mask * 1e-16
    distances = tf.sqrt(distances)
    distances = distances * (1.0 - mask)

    return distances


@tf.function
def triplet_loss(distances: tf.float32, mask_pos: tf.bool, mask_neg: tf.bool, margin: float) -> tf.float32:
    """Triplet Loss Function

    :param distances: pairwise distances of all embeddings within batch
    :param mask_pos: mask of distance between A and P (positive distances)
    :param mask_neg: mask of distances between A and N (negative distances
    :param margin: the margin for the triplet loss
        Formula: Loss = max(0, dist(A,P) - dist(A,N) + margin)
    :return: triplet loss values
    """

    pos_dists = tf.multiply(distances, tf.cast(mask_pos, tf.float32))
    hardest_pos_dists = tf.reduce_max(pos_dists, axis=1)
    neg_dists = tf.multiply(distances, tf.cast(mask_neg, tf.float32))
    neg_dists_max = tf.reduce_max(neg_dists, axis=1, keepdims=True)
    dists_manipulated = distances + neg_dists_max * (1.0 - tf.cast(mask_neg, tf.float32))
    hardest_neg_dist = tf.reduce_min(dists_manipulated, axis=1)

    return tf.maximum(hardest_pos_dists - hardest_neg_dist + margin, 0.0)


def OctupletLoss(margin: float = 0.5, metric: str = "euclidean", configuration: list = None):
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

    if configuration is None:
        configuration = [True, True, True, True]

    #@tf.function
    def _loss_function(embeddings: tf.float32, labels: tf.int64) -> tf.float32:
        """Octuplet Loss Function

        :param embeddings: concatenated high-resolution and low-resolution embeddings (size: 2*batch_size)
        :param labels: classes (size: batch_size)
        :return: loss value
        """

        batch_size = labels.shape[0]
        distances = pairwise_distances(embeddings, metric=metric)

        lbls_same = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        not_eye_bool = tf.logical_not(tf.cast(tf.eye(batch_size, batch_size), tf.bool))
        mask_pos = tf.equal(lbls_same, not_eye_bool)
        mask_neg = tf.logical_not(lbls_same)

        # TRIPLETS HR:HR ---------------------------------------------------------------
        dist_hrhr = tf.slice(distances, [0, 0], [batch_size, batch_size])
        loss_hrhr = triplet_loss(dist_hrhr, mask_pos, mask_neg, margin)

        # TRIPLETS HR:LR ---------------------------------------------------------------
        dist_hrlr = tf.slice(distances, [0, batch_size], [batch_size, batch_size])
        loss_hrlr = triplet_loss(dist_hrlr, mask_pos, mask_neg, margin)

        # TRIPLETS LR:HR ---------------------------------------------------------------
        dist_lrhr = tf.slice(distances, [batch_size, 0], [batch_size, batch_size])
        loss_lrhr = triplet_loss(dist_lrhr, mask_pos, mask_neg, margin)

        # TRIPLETS LR:LR ---------------------------------------------------------------
        dist_lrlr = tf.slice(distances, [batch_size, batch_size], [batch_size, batch_size])
        loss_lrlr = triplet_loss(dist_lrlr, mask_pos, mask_neg, margin)

        # Combination of triplet loss terms
        losses = tf.transpose(tf.cast([configuration] * batch_size, tf.float32)) * [
            loss_hrhr,
            loss_hrlr,
            loss_lrhr,
            loss_lrlr,
        ]
        summe = tf.reduce_sum(losses, axis=0)
        total_loss = tf.reduce_mean(summe, axis=0)

        return total_loss

    return _loss_function
