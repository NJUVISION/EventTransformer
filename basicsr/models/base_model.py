import logging
import os
from collections import OrderedDict
from copy import deepcopy

import torch
from ptflops import get_model_complexity_info
from pypushdeer import PushDeer
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only

logger = logging.getLogger('basicsr')


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def cal_flops_and_params(self, e_num):
        self.net_g.eval()
        macs, params = get_model_complexity_info(self.net_g, (e_num, 4), as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        flops = 2 * macs / 1024 / 1024 / 1024
        params = params / 1024 / 1024
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', f'{flops:.2f} G'))
        logger.info('{:<30}  {:<8}'.format('Number of parameters: ', f'{params:.2f} M'))
        logger.info('{:<30}  {:<8}'.format('Flops per events: ', f'{flops * 1024 / e_num:.2f} M/ev'))
        self.net_g.train()

    def feed_data(self, data):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def get_current_visuals(self):
        raise NotImplementedError

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        raise NotImplementedError

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=True, use_image=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
            rgb2bgr (bool): Whether to save images using rgb2bgr. Default: True
            use_image (bool): Whether to use saved images to compute metrics (PSNR, SSIM), if not, then use data directly from network' output. Default: True
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return self.nondist_validation(dataloader, current_iter, tb_logger,
                                           save_img, rgb2bgr, use_image)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        rank = os.environ['LOCAL_RANK']
        use_pbar = True if rank == '0' else False
        metric = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image,
                                         print_log=False, use_pbar=use_pbar)
        metric = torch.FloatTensor([metric])[0].to(int(rank))
        data_len = torch.FloatTensor([dataloader.__len__()])[0].to(int(rank))
        metric *= data_len
        torch.distributed.reduce(metric, dst=0)
        torch.distributed.reduce(data_len, dst=0)
        if rank == '0':
            metric = float(metric / data_len)
            dataset_name = dataloader.dataset.opt['name']
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            return metric
        else:
            return 0.

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        pushdeer = PushDeer(pushkey="PDU12888T5s2C2e3BMrym2vXKjHJSEHbUpTmRsnQx")
        pushdeer.send_text("EventTransformer", desp=log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def nondist_validation(self):
        raise NotImplementedError

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters',
                                                  False)
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def init_training_settings(self):
        self.net_g.train()
        # define losses
        self.setup_loss()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_loss(self):
        raise NotImplementedError

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(self.net_g.parameters(), **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(self.net_g.parameters(), **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'],
                                     save_filename)
            torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def load_pretrain_network(self):
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))
