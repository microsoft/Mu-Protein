import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np
import scipy.stats
import pathlib
import copy
import time
import random

import sys
sys.path.insert(0, os.path.join(Path(__file__).parent))
import vocab
from model import Muformer
from data import PairedDataset
from utils import Saver, EarlyStopping, Logger
import criterion
from fairseq import checkpoint_utils
from lr_scheduler import PolynomialLRDecay, InverseSqrtLRDecay
from esm import pretrained

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(local_rank):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ["RANK"])
        local_world_size = int(os.environ['WORLD_SIZE'])
        local_gpu = int(os.environ['LOCAL_RANK'])
        print('Get init distributed settings successfully, rank: {}, world_size: {}!'.format(local_rank,local_world_size))
    else:
        print('Error when get init distributed settings!')
        local_rank = local_rank
        local_world_size = torch.cuda.device_count()
        print('Use local setting, rank: {}, world_size: {}!'.format(local_rank,local_world_size))
        
    print('| distributed init (rank {}): env://'.format(local_rank), flush=True)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=local_world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)

    torch.distributed.barrier()
    setup_for_distributed(local_rank==0)
    return local_rank

def _init_weights(model_tmp):
    for m in model_tmp.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1e-2)     # 1e-5 # 1e-2
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Trainer(object):
    def __init__(self,
            output_dir=None,
            train_tsv=None,
            valid_tsv=None,
            test_tsv=None,
            fasta=None, 
            split_ratio=[0.9, 0.1],
            random_seed=42,
            num_ensembles=1,
            batch_size=128,
            hidden_size=256,
            dropout=0.1, 
            save_log=False,
            num_workers=1,
            aug_ratio=-1,
            encoder_name='pmlm',
            pretrained_model=None,
            loss='mae',
            lr=None,
            label_oper=None,
            decoder_name='mono',
            freeze_lm=False,
            local_rank=0,
            random_init=False,
            args=None,
            ):
        
        self.saver = Saver(output_dir=output_dir)
        self.logger = Logger(logfile = self.saver.save_dir / 'exp.log' if save_log else None)
        self.num_workers = num_workers

        self.logger.info(f'Pretrained: \t [{pretrained_model}]')
        
        if encoder_name == 'pmlm':
            tokendict_path = os.path.join(Path(__file__).parent, "protein/")

            arg_overrides = { 
                'data': tokendict_path
            }

            models, _, task = checkpoint_utils.load_model_ensemble_and_task(pretrained_model.split(os.pathsep), 
                                                                    arg_overrides=arg_overrides)
            encoder = models[0]
            alphabet = task.source_dictionary
            self.encoder_dim = encoder.args.encoder_embed_dim
            self.num_heads = encoder.args.encoder_attention_heads
        else:
            raise NotImplementedError()
            
        if random_init:
            _init_weights(encoder)

        self.encoder = encoder
        
        self.vocab_size = len(alphabet)

        self.logger.info(f'# vocab_size: {self.vocab_size}')

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.freeze_lm = freeze_lm
        
        self.num_ensembles = num_ensembles
        self.train_tsv = train_tsv

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            self.local_rank = init_distributed_mode(local_rank)

        self.args = args
        
        # set seed
        self.set_seed(seed=random_seed)

        self.label_column = getattr(args, "label_column", '')

        self.logger.info(f'Label column: \t [{self.label_column}]')
        
        self.dataset = PairedDataset(
            train_tsv=train_tsv, 
            valid_tsv=valid_tsv,
            test_tsv=test_tsv,
            fasta=fasta, 
            split_ratio=split_ratio,
            random_seed=random_seed,
            label_oper=label_oper,
            aug_ratio=aug_ratio,
            alphabet=alphabet,
            distributed=self.distributed,
            encoder_name=encoder_name,
            label_column=self.label_column,
            skip_bad_lines=args.skip_bad_lines)

        self.activate = getattr(self.args, 'activate', 'tanh')
        self.kernel_size = getattr(self.args, 'kernel_size', 3)
        self.conv_layers = getattr(self.args, 'conv_layers', 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f'Train data:  \t [{train_tsv}]')
        self.logger.info(f'Device: \t\t [{self.device}]')

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        self.criterion = criterion.get_criterion(loss)

        self.batch_size = batch_size
        
        self.encoder_lr = 1e-5
        self.decoder_lr = 1e-4

        self.lr = lr

        if lr:
            self.decoder_lr = lr
        
        if 'encoder_lr' in self.args:
            self.encoder_lr = self.args.encoder_lr
        if 'decoder_lr' in self.args:
            self.decoder_lr = self.args.decoder_lr
        
        self.weight_decay = getattr(self.args, 'weight_decay', 0.0)

        self.shuffle = getattr(self.args, 'shuffle', True)

        self.logger.info(f'Encoder LR:  \t [{self.encoder_lr}]')
        self.logger.info(f'Decoder LR:  \t [{self.decoder_lr}]')
        self.logger.info(f'Wei. Decay:  \t [{self.weight_decay}]')

        self.metric = getattr(self.args, 'metric', 'corr')

        self.clip = 1

        self.init_model()

        self._test_pack = None

    def init_model(self):
        muformer = Muformer(
            encoder=self.encoder, 
            vocab_size=self.vocab_size, 
            encoder_dim=self.encoder_dim, 
            hidden_size=self.hidden_size, 
            num_heads=self.num_heads, 
            dropout=self.dropout, 
            encoder_name=self.encoder_name, 
            decoder_name=self.decoder_name, 
            freeze_lm=self.freeze_lm, 
            activate=self.activate, 
            kernel_size=self.kernel_size, 
            conv_layers=self.conv_layers
        )

        if self.distributed:
            self.models = [torch.nn.parallel.DistributedDataParallel(muformer.to(self.device), device_ids=[self.local_rank],output_device=self.local_rank, find_unused_parameters=True) for _ in range(self.num_ensembles)]
        else:
            self.models = [torch.nn.parallel.DataParallel(muformer).to(self.device) for _ in range(self.num_ensembles)]
        
        self.optimizers = [optim.Adam([
            {'params': model.module.encoder.parameters(), 'lr': self.encoder_lr, 'weight_decay': self.weight_decay},
            {'params': model.module.decoder.parameters(), 'lr': self.decoder_lr, 'weight_decay': self.weight_decay},
        ]) for model in self.models]

    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        self.logger.info(f"Random seed: \t [{seed}]")

    @property
    def test_pack(self):
        if self._test_pack is None:
            test_loader, test_df = self.dataset.get_dataloader(
                'test', batch_size=self.batch_size, return_df=True, num_workers=self.num_workers)
            self._test_pack = (test_loader, test_df)
        return self._test_pack

    @property
    def test_loader(self):
        return self.test_pack[0]

    @property
    def test_df(self):
        return self.test_pack[1]

    def log(self, msg):
        if not self.distributed or self.local_rank == 0:
            self.logger.info(msg)

    def get_bare_model(self,net):
        if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            net = net.module
            return net
        else:
            return net
    
    def load_checkpoint(self, checkpoint_dir, load_optimizer=True):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise ValueError(f'[ ERRO ] {checkpoint_dir} is not a directory')
        for i in range(len(self.models)):
            checkpoint_path = f'{checkpoint_dir}/model_{i + 1}.pt'
            if self.distributed:
                if self.local_rank == 0:
                    self.logger.info('Load pretrained model from [{}]'.format(checkpoint_path))
            else:
                self.logger.info('Load pretrained model from [{}]'.format(checkpoint_path))

            # SECURITY: torch.load disabled to comply with the twin flames security incident (see microsoft/Mu-Protein#39).
            # Uncomment the line below only after confirming the checkpoint source is trusted.
            # pt = torch.load(checkpoint_path,map_location=torch.device('cpu'))
            print('[ WARNING ] torch.load is disabled due to the twin flames security incident; uncomment it in this file to restore loading behavior.')
            model_tmp = self.get_bare_model(self.models[i])
            model_dict = model_tmp.state_dict()

            incompatible_params = False
            for k, v in pt['model_state_dict'].items():
                if k not in model_dict:
                    self.logger.error(f'[ ERROR ] Saved but not used param: {k}')
                    incompatible_params = True
            for k, v in model_dict.items():
                if k not in pt['model_state_dict']:
                    self.logger.error(f'[ ERROR ] Used but not saved param: {k}')
                    incompatible_params = True
            
            if incompatible_params:
                raise Exception('The saved checkpoint is not compatible with the model.')

            model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
            model_dict.update(model_pretrained_dict)
            # self.models[i].load_state_dict(model_dict)
            model_tmp.load_state_dict(model_dict)
            if load_optimizer:
                self.optimizers[i].load_state_dict(pt['optimizer_state_dict'])

    def load_checkpoint_strip(self, checkpoint_dir, load_optimizer=True):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise ValueError(f'[ ERRO ] {checkpoint_dir} is not a directory')
        for i in range(len(self.models)):
            checkpoint_path = f'{checkpoint_dir}/model_{i + 1}.pt'
            if self.distributed:
                if self.local_rank == 0:
                    self.logger.info('Load pretrained model from [{}]'.format(checkpoint_path))
            else:
                self.logger.info('Load pretrained model from [{}]'.format(checkpoint_path))

            # SECURITY: torch.load disabled to comply with the twin flames security incident (see microsoft/Mu-Protein#39).
            # Uncomment the line below only after confirming the checkpoint source is trusted.
            # pt = torch.load(checkpoint_path,map_location=torch.device('cpu'))
            print('[ WARNING ] torch.load is disabled due to the twin flames security incident; uncomment it in this file to restore loading behavior.')
            model_tmp = self.get_bare_model(self.models[i])
            model_dict = model_tmp.state_dict()

            incompatible_params = False
            for k, v in pt['model_state_dict'].items():
                if k.replace('module.', '') not in model_dict:
                    self.logger.error(f'[ ERROR ] Saved but not used param: {k}')
                    incompatible_params = True
            for k, v in model_dict.items():
                if 'module.' + k not in pt['model_state_dict']:
                    self.logger.error(f'[ ERROR ] Used but not saved param: {k}')
                    incompatible_params = True
            
            if incompatible_params:
                raise Exception('The saved checkpoint is not compatible with the model.')

            model_pretrained_dict = {k.replace('module.', ''): v for k, v in pt['model_state_dict'].items() if k.replace('module.', '') in model_dict}
            model_dict.update(model_pretrained_dict)
            model_tmp.load_state_dict(model_dict)
            if load_optimizer:
                self.optimizers[i].load_state_dict(pt['optimizer_state_dict'])

    def load_pretrained_state(self, checkpoint_path, model=None, optimizer=None, is_resume=False):
        self.log(f'Load pretrained model from [{checkpoint_path}]')

        # SECURITY: torch.load disabled to comply with the twin flames security incident (see microsoft/Mu-Protein#39).
        # Uncomment the line below only after confirming the checkpoint source is trusted.
        # pt = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        print('[ WARNING ] torch.load is disabled due to the twin flames security incident; uncomment it in this file to restore loading behavior.')
        model_tmp = self.get_bare_model(model)
        model_dict = model_tmp.state_dict()
        model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
        model_dict.update(model_pretrained_dict)
        model_tmp.load_state_dict(model_dict)
        optimizer.load_state_dict(pt['optimizer_state_dict'])
        return (model, optimizer, pt['log_info']) if is_resume else (model, optimizer)
    
    def load_pretrained_model(self, checkpoint_dir, models):
        for idx, model in enumerate(models):
            # model = self.load_only_single_pretrained_model(os.path.join(checkpoint_dir,'model_{}.pt'.format(idx+1)),model)

            checkpoint_path = os.path.join(checkpoint_dir, f'model_{idx+1}.pt')
            self.log(f'Load pretrained model from [{checkpoint_path}]')

            # SECURITY: torch.load disabled to comply with the twin flames security incident (see microsoft/Mu-Protein#39).
            # Uncomment the line below only after confirming the checkpoint source is trusted.
            # pt = torch.load(checkpoint_path,map_location=torch.device('cpu'))
            print('[ WARNING ] torch.load is disabled due to the twin flames security incident; uncomment it in this file to restore loading behavior.')
            model_tmp = self.get_bare_model(model)
            model_dict = model_tmp.state_dict()
            model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
            model_dict.update(model_pretrained_dict)
            model_tmp.load_state_dict(model_dict)

        return models
    
    def save_checkpoint(self, ckp_name=None, model_dict=None, opt_dict=None, log_info=None):
        ckp = {'model_state_dict': model_dict, 'optimizer_state_dict': opt_dict}
        ckp['log_info'] = log_info
        self.saver.save_ckp(ckp, ckp_name)

    def train(self, epochs, eval_freq, patience, save_checkpoint, resume_path):  
        
        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers), start=1):

            (train_loader, train_df), (valid_loader, valid_df) = self._prepare_dataloaders()  

            steps_per_epoch = len(train_loader)

            warmup_epochs = getattr(self.args, 'warmup_epochs', 5)
            self.logger.info(f'#Warmup Epochs \t [{warmup_epochs}]')
            self.logger.info(f'Eval metric \t [{self.metric}]')
            self.logger.info(f'Batch size \t [{self.batch_size}]')
            self.logger.info(f'#Train_batches \t [{len(train_df)}]')
            self.logger.info(f'#Valid_batches \t [{len(valid_df)}]')

            start_epoch, best_score = self._resume_training_state(resume_path, model_idx, model, optimizer)  
            stopper = self._init_early_stopping(patience, eval_freq, best_score)  
            scheduler = self._init_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs) 

            best_model_state, best_optimizer_state_dict, best_epoch, best_val_results = None, None, None, None 

            for epoch in range(start_epoch, epochs + 1):
                time_start = time.time()

                train_loss = self._train_epoch(model_idx, model, optimizer, scheduler, train_loader, epoch)  

                if epoch % eval_freq == 0 or epoch == epochs:  
                    is_best, model_state, optimizer_state_dict, best_epoch, val_results = self._evaluate_and_save_best(  
                        model, optimizer, valid_loader, stopper, epoch, valid_df)
                    if is_best:
                        best_model_state = model_state
                        best_optimizer_state_dict = optimizer_state_dict
                        best_val_results = val_results

                    self._log_epoch_progress(train_loss, val_results, epoch, epochs, stopper, time.time() - time_start)  

                if stopper.early_stop or epoch == epochs:  
                    self._handle_early_stop(epoch, model_idx, best_model_state, best_optimizer_state_dict, best_epoch, best_val_results, stopper, save_checkpoint)  
                    break  

        self._finalize_training()  

    def _prepare_dataloaders(self):  
        (train_loader, train_df), (valid_loader, valid_df) = \
            self.dataset.get_dataloader(
                'train_valid', self.batch_size,
                return_df=True, resample_train_valid=False, num_workers=self.num_workers)
        if self.distributed:  
            self.dataset.sampler_train.set_epoch(0)
        return (train_loader, train_df), (valid_loader, valid_df)

    def _resume_training_state(self, resume_path, model_idx, model, optimizer):  
        start_epoch = 1  
        best_score = None  

        if resume_path is not None:  
            checkpoint_path = f'{resume_path}/model_{model_idx}.pt'  
            self.logger.info(f'Loading checkpoint from {checkpoint_path}')  
            # SECURITY: torch.load disabled to comply with the twin flames security incident (see microsoft/Mu-Protein#39).
            # Uncomment the line below only after confirming the checkpoint source is trusted.
            # checkpoint = torch.load(checkpoint_path, map_location=self.device)  
            print('[ WARNING ] torch.load is disabled due to the twin flames security incident; uncomment it in this file to restore loading behavior.')

            model.load_state_dict(checkpoint['model_state_dict'])  
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
            start_epoch = checkpoint.get('epoch', start_epoch) + 1  
            best_score = checkpoint.get(f'best_{self.metric}', best_score)  

            self.logger.info(f'Resumed training from epoch {start_epoch} with best {self.metric}: {best_score}')  

        return start_epoch, best_score

    def _init_early_stopping(self, patience, eval_freq, best_score): 
        higher_better = True if self.metric in ['accuracy', 'corr'] else False  
        return EarlyStopping(patience=patience, eval_freq=eval_freq, best_score=best_score, higher_better=higher_better)  

    def _init_scheduler(self, optimizer, epochs, steps_per_epoch, warmup_epochs):
        return InverseSqrtLRDecay(optimizer, warmup_updates=warmup_epochs*steps_per_epoch)  
        # return PolynomialLRDecay(optimizer, total_num_step=epochs*steps_per_epoch, end_learning_rate=1e-8, power=2.0, warmup_updates=warmup_epochs*steps_per_epoch)
        # return PolynomialLRDecay(optimizer, total_num_step=epochs, end_learning_rate=1e-8, power=2.0, warmup_updates=warmup_epochs)

    def _train_epoch(self, midx, model, optimizer, scheduler, train_loader, epoch):  
        
        model.train()

        ### --- Shuffling training data ---
        (train_loader, train_df), (valid_loader, valid_df) = \
        self.dataset.get_dataloader(
            'train_valid', self.batch_size,
            return_df=True, resample_train_valid=False, num_workers=self.num_workers, shuffle=self.shuffle)
        ### -------------------------------

        total_loss = 0
        for step, batch in tqdm(enumerate(train_loader, 1),leave=False, desc=f'M{midx} E{epoch}', total=len(train_loader), ncols=80):
            optimizer.zero_grad()

            rec1, rec2 = batch[0], batch[1]
            x1, x2 = rec1['x'].to(self.device), rec2['x'].to(self.device)
            x1_mask, x2_mask = rec1['mask'].to(self.device), rec2['mask'].to(self.device)

            output = model(x1, x2, x1_mask=x1_mask, x2_mask=x2_mask)
                
            y_real = rec1['label_af'].to(self.device)
            loss = self.criterion(output.view(-1), y_real)
            
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            optimizer.step()
            scheduler.step() 
            
            total_loss += loss.item()
            
        return total_loss / step
        
    def _evaluate_and_save_best(self, model, optimizer, valid_loader,  
                                stopper, epoch, valid_df):  
        val_results = self.test(test_model=model, test_loader=valid_loader,
                test_df=valid_df, mode='val')

        is_best = stopper.update(val_results['metric'][self.metric])

        if is_best:
            best_model_state_dict = copy.deepcopy(model.module.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch
        else:
            best_model_state_dict, best_optimizer_state_dict, best_epoch = None, None, None
        
        return is_best, best_model_state_dict, best_optimizer_state_dict, best_epoch, val_results

    def _log_epoch_progress(self, train_loss, val_results, epoch, epochs, stopper, delta_time):  
        self.log(f'E: {epoch}/{epochs} | train loss: {train_loss:.4f} | valid loss: {val_results["loss"]:.4f} | ' 
                + ' | '.join([f'valid {k}: {v:.4f}' for (k, v) in val_results['metric'].items() if k!='loss'])
            + f' | best valid {self.metric}: {stopper.best_score:.4f} | {delta_time:.1f} s/epoch')

    def _handle_early_stop(self, epoch, model_idx, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_val_results, stopper, save_checkpoint):  
        self.log(f'Stopped at epoch {epoch}')
        
        if save_checkpoint:
            if not self.distributed or self.local_rank == 0:
                self.save_checkpoint(ckp_name=f'model_{model_idx}.pt',
                    model_dict=best_model_state_dict,
                    opt_dict=best_optimizer_state_dict,
                    # opt_dict = optimizer.state_dict(),
                    log_info={
                        'epoch': best_epoch,
                        f'best_{self.metric}': stopper.best_score,
                        'val_loss': best_val_results['loss'],
                        'val_results': best_val_results['metric']
                    })
            
            if self.distributed:
                torch.distributed.barrier()

    def _finalize_training(self):  
        if self.distributed:
            torch.distributed.barrier() 

    def test(self, test_model=None, test_loader=None, test_df=None,
                checkpoint_dir=None, save_prediction=False, mode='test'):
        
        if checkpoint_dir is not None:
            self.models = self.load_pretrained_model(checkpoint_dir, self.models)
        
        if test_loader is None and test_df is None:
            test_loader = self.test_loader
            test_df = self.test_df
        
        test_models = self.models if test_model is None else [test_model]
        esb_ypred = None
        esb_loss = 0

        for model in test_models:
            model.eval()
            y_pred = None
            tot_loss = 0
            y_fitness = torch.Tensor([])
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_loader, 1), desc=mode, leave=False, total=len(test_loader), ncols=100):
                    rec1, rec2 = batch[0], batch[1]
                    x1, x2 = rec1['x'].to(self.device), rec2['x'].to(self.device)
                    x1_mask, x2_mask = rec1['mask'].to(self.device), rec2['mask'].to(self.device)

                    y_fitness = torch.cat([y_fitness, rec1['label_af']], dim=-1)

                    output = model(x1, x2, x1_mask=x1_mask, x2_mask=x2_mask)
                        
                    y = rec1['label_af'].to(self.device)
                    loss = self.criterion(output.view(-1), y)
                    tot_loss += loss.item()

                    y_pred = output if y_pred is None else torch.cat((y_pred, output), dim=0)

            y_pred = y_pred.detach().cpu() if self.device == torch.device('cuda') else y_pred.detach()
            esb_ypred = y_pred.view(-1, 1) if esb_ypred is None else torch.cat((esb_ypred, y_pred.view(-1, 1)), dim=1)
            esb_loss += tot_loss / step

        esb_ypred = esb_ypred.mean(axis=1).numpy()
        esb_loss /= len(test_models)

        y_fitness = y_fitness.numpy()
        eval_results = scipy.stats.spearmanr(y_fitness, esb_ypred)[0]

        test_results = {}
        if mode in ['test', 'ensemble']:
            results_df = test_df.copy()
            results_df['prediction'] = esb_ypred
            test_results['df'] = results_df
            if save_prediction:
                self.saver.save_df(results_df, 'prediction.tsv')
        test_results['loss'] = esb_loss
        test_results['metric'] = {'corr': eval_results, 'loss': esb_loss}

        return test_results 