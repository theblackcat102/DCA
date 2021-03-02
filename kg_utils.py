import torch
import numpy as np
import torch.nn.functional as F

def diversity_regularization(x):
    x = x / F.normalize(x, dim=-1, p=2)
    y = torch.flip(x, [0, 1])

    return torch.cdist(x, y, p=2).mean()



def evaluate(model, batch, t, hit_k=10):
    triplet, _ = batch
    h, r  = triplet[:2]
    pred_ranks = model.predict(h, r)

    hits = [[]]*10

    for (t_, pred_rank) in zip(t, pred_ranks):

        for idx in range(1,11):
            hits[idx-1].append((pred_rank[:idx] == t_).sum())

    for idx in range(10):
        hits[idx] = np.sum(hits[idx])

    return np.array(hits), len(t)

def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, k: int = 10) -> int:
    """Calculates number of hits@k.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :param device: device on which calculations are taking place
    :param k: number of top K results to be considered as hits
    :return: Hits@K score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    zero_tensor = torch.tensor([0])
    one_tensor = torch.tensor([1])
    
    _, indices = predictions.topk(k=k, largest=False)
    if indices.is_cuda:
        indices = indices.cpu()
    if ground_truth_idx.is_cuda:
        ground_truth_idx = ground_truth_idx.cpu()

    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()


def mrr(predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
    """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :return: Mean reciprocal rank score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    indices = predictions.argsort()
    return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()


def chunks(lst, n):
    for idx in range(0, len(lst), n):
        yield lst[idx:idx+n]

def evaluate_(model, batch, hits_k=10):
    triplet = batch[0]
    h, r, t  = triplet
    # torch.arange(end=entities_count, device=device)
    batch_size = h.size()[0]
    
    if model.num_entities < 1e6:
        entity_ids = torch.arange(end=model.num_entities)
        if h.is_cuda:
            entity_ids = entity_ids.cuda()
        all_entities = entity_ids.repeat(batch_size, 1)
        heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails_predictions =  model.score(*model(heads, relations, all_entities))

        heads_predictions =  model.score(*model(all_entities, relations, tails))

        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((t.reshape(-1, 1), h.reshape(-1, 1)))
        if predictions.is_cuda:
            predictions = predictions.cpu()
        if ground_truth_entity_id.is_cuda:
            ground_truth_entity_id = ground_truth_entity_id.cpu()

        hits_score = [0]*hits_k
        for hit_k in range(hits_k):
            hits_score[hit_k] = hit_at_k(predictions, ground_truth_entity_id, k=hit_k)
        mrr_score = mrr(predictions, ground_truth_entity_id)
        # triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        return hits_score, mrr_score, batch_size
    else:
        all_entity_ids = torch.arange(end=model.num_entities)
        ground_truth_entity_ids = None
        tails_predictions = None
        heads_predictions = None

        for entity_ids in chunks(all_entity_ids, 50000):
            if h.is_cuda:
                entity_ids = entity_ids.cuda()
            all_entities = entity_ids.repeat(batch_size, 1)
            heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
            relations = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])

            tails_prediction =  model.score(*model(heads, relations, all_entities))
            heads_prediction =  model.score(*model(all_entities, relations, tails))

            if heads_prediction.is_cuda:
                heads_prediction = heads_prediction.cpu()

            if heads_predictions is None:
                heads_predictions = heads_prediction
            else:
                heads_predictions = torch.cat([heads_predictions, heads_prediction], dim=1)

            if tails_prediction.is_cuda:
                tails_prediction = tails_prediction.cpu()

            if tails_predictions is None:
                tails_predictions = tails_prediction
            else:
                tails_predictions = torch.cat([tails_predictions, tails_prediction], dim=1)


        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((t.reshape(-1, 1), h.reshape(-1, 1)))

        if ground_truth_entity_id.is_cuda:
            ground_truth_entity_id = ground_truth_entity_id.cpu()

        hits_score = [0]*hits_k
        for hit_k in range(hits_k):
            hits_score[hit_k] = hit_at_k(predictions, ground_truth_entity_id, k=hit_k)
        mrr_score = mrr(predictions, ground_truth_entity_id)
        # triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        return hits_score, mrr_score, batch_size


import pytorch_lightning as pl
import os

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="step_checkpoint",
        total_checkpoint=5,
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.checkpoints = []
        self.total_checkpoint = total_checkpoint

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            self.checkpoints.append(ckpt_path)
            trainer.save_checkpoint(ckpt_path)
        while len(self.checkpoints) > self.total_checkpoint:
            prev_ckpt_path = self.checkpoints.pop(0)
            try:
                if os.path.exists(prev_ckpt_path):
                    os.remove(prev_ckpt_path)
            except FileNotFoundError:
                break