import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sonar_otlr.model import SATML, TailMemory, open_loss, synthesize


class SonarImages(Dataset):
    def __init__(self, root, rows, size):
        self.root, self.rows = root, rows
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, label = self.rows[index]
        with Image.open(self.root / name) as image:
            return self.transform(image.convert('RGB')), label


def split(root, seed, class_caps=None):
    known = ['floats', 'tire', 'cylinder', 'big_propeller', 'fishing_net']
    rng = random.Random(seed)
    seen = set()
    rows = {k: [] for k in ('train', 'validation', 'test')}
    counts = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        paths = sorted(p for p in folder.glob('*.jpg') if p.is_file())
        rng.shuffle(paths)
        label = known.index(folder.name) if folder.name in known else -1
        unique = []
        for path in paths:
            with Image.open(path) as image:
                pixels = image.convert('RGB')
                digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).digest()
            if digest not in seen:
                unique.append((path.relative_to(root).as_posix(), label))
                seen.add(digest)
        if label == -1:
            rows['test'].extend(unique[:45])
        else:
            holdout = min(30, len(unique) // 5)
            if not holdout:
                raise ValueError(f'Not enough samples for {folder.name}')
            rows['validation'].extend(unique[:holdout])
            rows['test'].extend(unique[holdout:2 * holdout])
            rows['train'].extend(unique[2 * holdout:2 * holdout + (class_caps or {}).get(folder.name, 128)])
    counts = [sum(label == c for _, label in rows['train']) for c in range(len(known))]
    if min(counts) == 0 or not any(y == -1 for _, y in rows['test']):
        raise ValueError('Missing known or unknown classes')
    return known, counts, rows


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).cpu() for x, _ in loader])


def evaluate(scores, labels, threshold, classes):
    confidence, prediction = scores.max(1)
    known = labels >= 0
    correct = prediction.eq(labels)
    accepted = confidence >= threshold
    # Group tied scores before integrating macro CCR against unknown FPR.
    recall_weights = torch.zeros_like(confidence)
    for c in range(classes):
        cls = labels == c
        recall_weights[cls] = 1 / (classes * cls.sum())
    order = confidence.argsort(descending=True)
    sorted_conf = confidence[order]
    boundaries = torch.cat((sorted_conf[:-1] != sorted_conf[1:], torch.tensor([True])))
    ccr = (recall_weights * correct)[order].cumsum(0)[boundaries]
    fpr = (~known)[order].float().cumsum(0)[boundaries] / (~known).sum()
    area = torch.trapz(F.pad(ccr, (1, 0)), F.pad(fpr, (1, 0)))
    return dict(known_top1=correct[known].float().mean().item(),
                known_correct_accepted=(correct & accepted)[known].float().mean().item(),
                unknown_recall=(~accepted)[~known].float().mean().item(),
                oscr_macro=area.item(), threshold=threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/nksid_quick.json')
    parser.add_argument('--evaluate', type=Path, help='Checkpoint to retest with its saved split and threshold')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.evaluate, map_location='cpu', weights_only=True) if args.evaluate else None
    if state:
        cfg = state['config']
    data = ROOT / cfg['data']
    classes, counts, rows = ((state['classes'], state['counts'], state['split'])
                             if state else split(data, cfg['seed'], cfg.get('class_caps')))
    def loader(key, shuffle=False):
        return DataLoader(SonarImages(data, rows[key], cfg['size']), batch_size=cfg['batch'],
                          shuffle=shuffle, drop_last=shuffle,
                          generator=torch.Generator().manual_seed(cfg['seed']))
    labels = torch.tensor([y for _, y in rows['test']])
    if state:
        model = SATML(len(classes)).to(device)
        model.load_state_dict(state['model'])
        result = evaluate(predict(model, loader('test'), device), labels, state['threshold'], len(classes))
        print(json.dumps(result, indent=2))
        return
    out, checkpoint = ROOT / cfg['output'], ROOT / cfg['checkpoints']
    out.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    (out / 'split.json').write_text(json.dumps(rows, indent=2))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    results = {}
    defaults = [
        dict(name=name, sapfs=sapfs, ptmcl=ptmcl)
        for name, sapfs, ptmcl in [('baseline', False, False), ('sapfs', True, False),
                                 ('ptmcl', False, True), ('satml', True, True)]
    ]
    for variant in cfg.get('variants', defaults):
        name, sapfs, ptmcl = (variant[key] for key in ('name', 'sapfs', 'ptmcl'))
        r_min, r_max = variant.get('r_min', .2), variant.get('r_max', .8)
        rho, momentum = variant.get('rho', .2), variant.get('momentum', .9)
        weight = variant.get('weight', 3.)
        if not (0 <= r_min <= r_max <= 1 and 0 <= rho <= 1 and 0 <= momentum < 1 and weight >= 0):
            raise ValueError(f'Invalid variant parameters: {variant}')

        torch.manual_seed(cfg.get('train_seed', cfg['seed']))
        model = SATML(len(classes), ROOT / cfg['weights']).to(device)
        memory = TailMemory(counts, cfg['tail_minimum'], rho, momentum) if ptmcl else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        train_loader = loader('train', True)
        if not len(train_loader):
            raise ValueError('Training split smaller than one batch')
        history = []
        steps = 0
        intermediate = {}
        milestones = set(cfg.get('evaluation_steps', []))
        validation_loader, test_loader = loader('validation'), loader('test')

        def score_current():
            validation_conf = predict(model, validation_loader, device).max(1).values
            threshold = validation_conf.sort().values[int(.05 * len(validation_conf))].item()
            scores = predict(model, test_loader, device)
            return scores, evaluate(scores, labels, threshold, len(classes))

        for epoch in range(cfg['epochs']):
            model.train()
            totals = torch.zeros(3, device=device)
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                half = len(images) // 2
                maps = model.spatial(images)
                known_maps, y = maps[half:], labels_batch[half:]
                logits = model.classify(known_maps)
                ce = F.cross_entropy(logits, y)
                opened = ce.new_zeros(())
                contrast = ce.new_zeros(())
                if sapfs:
                    pseudo = synthesize(maps[:half], labels_batch[:half], model.net.fc.weight, r_min, r_max,
                                        variant.get("strategy", "gini"))
                    opened = open_loss(logits, y, model.classify(pseudo))
                if memory:
                    contrast = memory.loss(known_maps.mean((-2, -1)), y)
                loss = ce + opened + weight * contrast
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{name}: nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                if steps in milestones:
                    _, intermediate[steps] = score_current()
                    model.train()

                totals += torch.stack((ce, opened, contrast)).detach()
            history.append((totals / len(train_loader)).tolist())
            print(name, epoch + 1, history[-1], flush=True)
        scores, metrics = score_current()
        threshold = metrics['threshold']
        results[name] = dict(**metrics, parameters=variant, loss_history=history,
                             optimizer_steps=steps, intermediate=intermediate,
                             tail_classes=[classes[c] for c in memory.tail] if memory else [],
                             selected_memories=memory.selected if memory else 0,
                             active_contrast_steps=memory.active_steps if memory else 0)
        if cfg.get('save_checkpoints', True) and variant.get('save_checkpoint', True):
            torch.save(dict(model=model.state_dict(), config=cfg, split=rows, classes=classes,
                        counts=counts, threshold=threshold), checkpoint / f'{name}.pt')
        (out / f'{name}_predictions.json').write_text(json.dumps(scores.tolist()))
        (out / 'results.json').write_text(json.dumps(dict(device=str(device), train_counts=dict(zip(classes, counts)),
                split_sizes={k: len(v) for k,v in rows.items()}, variants=results), indent=2))
        print(name, metrics, flush=True)


if __name__ == '__main__':
    main()
