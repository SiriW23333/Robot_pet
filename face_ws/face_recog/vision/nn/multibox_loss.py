import numpy as np


class MultiboxLoss:
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
        and Smooth L1 regression loss.
        """
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = np.array(priors)
        self.device = device

    def log_softmax(self, x, axis=-1):
        """Compute log softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        x_stable = x - x_max
        log_sum_exp = np.log(np.sum(np.exp(x_stable), axis=axis, keepdims=True))
        return x_stable - log_sum_exp

    def cross_entropy(self, predictions, targets):
        """Compute cross-entropy loss."""
        num_samples = predictions.shape[0]
        log_probs = -np.log(predictions[np.arange(num_samples), targets])
        return np.sum(log_probs)

    def smooth_l1_loss(self, predictions, targets):
        """Compute Smooth L1 loss."""
        diff = np.abs(predictions - targets)
        loss = np.where(diff < 1, 0.5 * diff ** 2, diff - 0.5)
        return np.sum(loss)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding to all the priors.
        """
        num_classes = confidence.shape[2]

        # Compute hard negative mining mask
        loss = -self.log_softmax(confidence, axis=2)[:, :, 0]
        mask = self.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # Classification loss
        confidence = confidence[mask, :]
        classification_loss = self.cross_entropy(confidence, labels[mask])

        # Smooth L1 loss
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = self.smooth_l1_loss(predicted_locations, gt_locations)

        num_pos = gt_locations.shape[0]
        return smooth_l1_loss / num_pos, classification_loss / num_pos

    def hard_negative_mining(self, loss, labels, neg_pos_ratio):
        """Perform hard negative mining."""
        pos_mask = labels > 0
        num_pos = np.sum(pos_mask, axis=1, keepdims=True)
        num_neg = np.clip(num_pos * neg_pos_ratio, 0, labels.shape[1] - 1)

        # Sort losses for negative samples
        loss[pos_mask] = -np.inf  # Exclude positive samples
        sorted_idx = np.argsort(-loss, axis=1)
        rank = np.argsort(sorted_idx, axis=1)

        neg_mask = rank < num_neg
        return pos_mask | neg_mask