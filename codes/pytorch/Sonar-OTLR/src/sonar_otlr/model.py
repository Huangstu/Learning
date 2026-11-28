from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class SATML(nn.Module):
    def __init__(self, classes, weights=None):
        super().__init__()
        self.net = resnet18()
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.net.fc = nn.Linear(512, classes)

    def spatial(self, x):
        for layer in list(self.net.children())[:-2]:
            x = layer(x)
        return x

    def classify(self, maps):
        return self.net.fc(maps.mean((-2, -1)))

    def forward(self, x):
        return self.classify(self.spatial(x))


def synthesize(maps, labels, weight, r_min=.2, r_max=.8, strategy="gini"):
    with torch.no_grad():
        cam = (maps * weight[labels, :, None, None]).sum(1).relu().flatten(1)
        cam = (cam - cam.amin(1, keepdim=True))
        cam = cam / cam.amax(1, keepdim=True).clamp_min(1e-8)
        ordered = cam.sort(dim=1).values
        length = ordered.size(1)
        rank = torch.arange(1, length + 1, device=maps.device)
        gini = ((2 * (ordered * rank).sum(1) / (length * ordered.sum(1) + 1e-8))
                - (length + 1) / length).clamp(0, 1)
        if strategy == "gini":
            sparsity = gini
        elif strategy == "entropy":
            probability = cam / cam.sum(1, keepdim=True).clamp_min(1e-8)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
            sparsity = (1 - entropy / torch.tensor(float(length)).log().item()).clamp(0, 1) if length > 1 else gini
            sparsity = torch.where(cam.sum(1) > 0, sparsity, gini)
        else:
            raise ValueError(f"Unknown sparsity strategy: {strategy}")
        keep = r_min + (r_max - r_min) * sparsity
        index = (1 - keep) * (length - 1)
        lo, hi = index.floor().long(), index.ceil().long()
        rows = torch.arange(len(maps), device=maps.device)
        threshold = ordered[rows, lo].lerp(ordered[rows, hi], index - lo)
        mask = (cam * (cam >= threshold[:, None])).reshape(len(maps), 1, *maps.shape[-2:])
        order = torch.randperm(len(maps), device=maps.device)
        partner = torch.empty_like(order)
        partner[order] = order.roll(1)
    return mask * maps + (1 - mask) * maps[partner]


def open_loss(known, labels, pseudo, gamma=.5):
    positive = F.one_hot(labels, known.shape[1]).bool()
    return ((gamma * pseudo).sigmoid().mean()
            + (gamma * known.masked_fill(positive, -torch.inf)).sigmoid().mean()
            + (-gamma * known[positive]).sigmoid().mean())


class TailMemory:
    def __init__(self, counts, minimum, rho=.2, momentum=.9):
        self.momentum = momentum
        threshold = max(torch.tensor(counts, dtype=torch.float).quantile(rho).item(), minimum)
        self.tail = [i for i, count in enumerate(counts) if count <= threshold]
        self.features = {i: deque(maxlen=32) for i in self.tail}
        self.history = {i: deque(maxlen=32) for i in self.tail}
        self.prototypes = {}
        self.selected = 0
        self.active_steps = 0

    def loss(self, features, labels):
        z = F.normalize(features, dim=1)
        candidates, targets = [z], [labels]
        with torch.no_grad():
            for c in labels.unique().tolist():
                center = z[labels == c].mean(0)
                previous = self.prototypes.get(c)
                self.prototypes[c] = F.normalize(center if previous is None else self.momentum * previous + (1 - self.momentum) * center, dim=0)
            for c in self.tail:
                queue = self.features[c]
                if not queue:
                    continue
                bank = torch.stack(tuple(queue))
                score = bank @ self.prototypes[c]
                history = self.history[c]
                threshold = max(.1, torch.tensor(tuple(history)).quantile(.2).item()) if len(history) >= 5 else .1
                history.extend(score.tolist())
                good = bank[score >= threshold]
                good = good[torch.randperm(len(good), device=z.device)[:4]]
                if len(good):
                    candidates.append(good)
                    targets.append(labels.new_full((len(good),), c))
                    self.selected += len(good)
            for c in self.tail:
                self.features[c].extend(z.detach()[labels == c].unbind())
        similarity = z @ torch.cat(candidates).T / .07
        self_mask = torch.eye(len(z), similarity.shape[1], device=z.device, dtype=torch.bool)
        positive = labels[:, None].eq(torch.cat(targets)) & ~self_mask
        log_prob = similarity - similarity.masked_fill(self_mask, -torch.inf).logsumexp(1, keepdim=True)
        count = positive.sum(1)
        valid = count > 0
        if not valid.any():
            return z.sum() * 0
        self.active_steps += 1
        loss = log_prob.masked_fill(~positive, 0).sum(1).neg() / count.clamp_min(1)
        return torch.stack([loss[valid & (labels == c)].mean() for c in labels[valid].unique()]).mean()
