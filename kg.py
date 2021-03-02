import pytorch_lightning as pl
from dataset import KGDataset
from kg_utils import evaluate_, CheckpointEveryNSteps
import torch
import numpy as np
import sys, os
import torch.nn.functional as F
from kg_models import TransE, DistMult
from torch.optim.lr_scheduler import LambdaLR
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("name", None, "knowledge graph name")
flags.DEFINE_string("model_name", 'transe', "knowledge graph type")

flags.DEFINE_string("output_dir", None,
                     "knowledge graph output directory")
flags.DEFINE_integer("gpus", 1,
                     "number of gpus to use")
flags.DEFINE_float("lr", 0.01,
                     "learning rate")
flags.DEFINE_float("l2", 7.469e-12,
                     "L2 regularization weight")
flags.DEFINE_float("self_regul", 0,
                     "self regularization")
flags.DEFINE_float("diversity", -1e-12,
                     "diversity regularization")

flags.DEFINE_integer("accum_batch", 1,
                     "gradient batch accumulation")
flags.DEFINE_integer("num_workers", 8,
                     "dataloader workers")
flags.DEFINE_string("optimizer", 'sgd',
                     "optimizer type")

flags.DEFINE_integer("epochs", 1000,
                     "training epoch")

flags.DEFINE_integer("dimension", 300,
                     "dimension value")
flags.DEFINE_integer("val_batch_size", 2,
                     "validation batch size")
flags.DEFINE_integer("train_batch_size", 128,
                     "training batch size")
flags.DEFINE_string("backend", None,
                     "distributed backend None, dp, ddp")
flags.DEFINE_boolean('mean_loss', True, 'Include mean of type loss')

flags.DEFINE_boolean('eval', False, 'Evaluate only')

flags.DEFINE_string("ckpt", None,
                     "checkpoint path")

flags.mark_flag_as_required("name")
flags.mark_flag_as_required("output_dir")



FLAGS(sys.argv)

def diversity_regularization(x):
    x = x / F.normalize(x, dim=-1, p=2)
    y = torch.flip(x, [0, 1])

    return torch.cdist(x, y, p=2).mean()

class KGTrainer(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(*batch)

    def training_step(self, batch, batch_nb):
        pos_triplets, neg_triplets, type_triplets = batch

        loss1 = self.model.calculate_loss(pos_triplets, neg_triplets)
        results = {}
        pos_rels = pos_triplets[1]
        results['loss'] = loss1

        if FLAGS.mean_loss:
            loss2, diversity_loss, l2_regularization = self.model.calculate_loss_avg(type_triplets)
            results['avg_loss'] = loss2
            results['diversity_loss'] = diversity_loss
            results['l2_regularization'] = l2_regularization

            loss = (loss1 + loss2 * 0.001 )
            if FLAGS.diversity > 0:
                loss +=  diversity_loss * (FLAGS.diversity)

            if FLAGS.l2 > 0:
                loss = loss + l2_regularization * FLAGS.l2
        else:
            loss = loss1
        if FLAGS.diversity > 0:
            div_regularization_loss = diversity_regularization(self.model.rel_embeddings(pos_rels))
            results['div_regularization_loss'] = div_regularization_loss

            loss = loss +  div_regularization_loss * (FLAGS.diversity)

        if FLAGS.l2 > 0:
            self_regularization = self.model.self_regularization()
            results['l2'] = self_regularization

            loss = loss + self_regularization * FLAGS.l2

        if FLAGS.self_regul > 0:
            pos_self_regul_loss = self.model.regularization(pos_triplets)
            results['pos_self_regul_loss'] = pos_self_regul_loss
            loss += pos_self_regul_loss * FLAGS.self_regul

        tensorboard_logs = { 'loss': loss, 'self_regularization': self_regularization,  }

        if FLAGS.mean_loss:
            tensorboard_logs['diversity_penalty'] = diversity_loss
            tensorboard_logs['avg_loss'] = loss2

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        pos_triplets, neg_triplets, _ = batch

        loss = self.model.calculate_loss(pos_triplets, neg_triplets)

        l2_regularization =  self.model.self_regularization()

        loss = loss + l2_regularization * FLAGS.l2

        if FLAGS.self_regul != 0:
            loss += self.model.regularization(pos_triplets) * FLAGS.self_regul


        hits, mrr, cnt = evaluate_(self.model, batch)

        return {'val_loss': loss, 'cnt': cnt, 'hits': hits, 'mrr': mrr}

    def validation_epoch_end(self, outputs):

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': val_loss_mean}
        total_cnt = np.sum([ x['cnt']  for x in outputs ])
        log['val/mrr'] = np.sum([ x['mrr']  for x in outputs ]) / total_cnt
        log['val/hit@3'] = np.sum([ x['hits'][2]  for x in outputs ]) / total_cnt
        log['val/hit@10'] = np.sum([ x['hits'][9]  for x in outputs ]) / total_cnt

        return {'val_loss': val_loss_mean, 'log': log}

    def configure_optimizers(self):
        if FLAGS.optimizer == 'sgd':
            optimizer_cls = torch.optim.SGD
        elif FLAGS.optimizer == 'adam':
            optimizer_cls = torch.optim.Adam
        elif FLAGS.optimizer == 'adagrad':
            optimizer_cls = torch.optim.Adagrad
        optimizer = optimizer_cls(self.model.parameters(), lr=FLAGS.lr)

        def get_linear_warmup(optimizer, num_warmup_steps, last_epoch=-1):
            def lr_lambda(current_step):
                learning_rate = min(1.0, float(current_step) / float(num_warmup_steps))
                return learning_rate

            return LambdaLR(optimizer, lr_lambda, last_epoch)

        self.lr_scheduler = get_linear_warmup(optimizer, num_warmup_steps=1000)

        return [optimizer], [{ 'scheduler': self.lr_scheduler, 'name': 'linear_warmup','interval': 'step', }]


if __name__ == "__main__":
    # dataset = SPO('kgs/HowNet.spo', 'hownet')
    train_dataset = KGDataset()
    valid_dataset = train_dataset

    train = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.train_batch_size, 
        num_workers=FLAGS.num_workers, shuffle=True)

    valid = None
    if valid_dataset != None:
        valid = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.val_batch_size, 
            num_workers=FLAGS.num_workers)

    entity_size, rel_size, type_size = train_dataset.entity_size, train_dataset.relation_size, train_dataset.type_size

    model_class = TransE

    # pad and mask
    model = model_class( entity_size+2, rel_size+1, type_size+1, FLAGS.dimension )

    model_wrapper = KGTrainer(model)
    trainer = pl.Trainer(gpus=FLAGS.gpus, max_epochs=FLAGS.epochs,
        distributed_backend=FLAGS.backend,
        # precision=16,
        # amp_level='O3',
        num_sanity_val_steps=5,
        accumulate_grad_batches=FLAGS.accum_batch,
        default_root_dir=FLAGS.output_dir,
        check_val_every_n_epoch=10, 
        limit_val_batches=0.02, # limit_val_batches, val_percent_check
        callbacks=[
            CheckpointEveryNSteps(
                save_step_frequency=15000,
                total_checkpoint=5,
            ),
        ])

    trainer.fit(model_wrapper, train, val_dataloaders=valid)
