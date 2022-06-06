from torch.nn import (
    Module,
    Sequential,
    Linear,
    LeakyReLU,
    Conv3d,
    BatchNorm3d,
    ConvTranspose3d,
    BatchNorm1d,
    Tanh
)
import torch
import numpy as np
from torch import split, exp, randn_like
import torch.nn.functional as F
import sys
sys.path.append("/user/fit2form")
from utils import Flatten
import tqdm


class MAML_Encoder(Module):
    def __init__(self, embedding_dim, max_log_var=0.1):
        super(MAML_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_log_var = max_log_var
        self.emb_range_limit = Tanh()
        self.net = Sequential(
            Conv3d(1, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            Conv3d(8, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            Conv3d(16, 32, kernel_size=3, stride=1),
            BatchNorm3d(32),
            LeakyReLU(),
            Conv3d(32, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            Conv3d(64, 64, kernel_size=3, stride=2),
            BatchNorm3d(64),
            LeakyReLU(),
            Conv3d(64, 32, kernel_size=3, stride=1),
            BatchNorm3d(32),
            LeakyReLU(),
            Conv3d(32, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            Conv3d(16, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            Conv3d(8, 4, kernel_size=3, stride=1),
            BatchNorm3d(4),
            LeakyReLU(),
            Flatten()
        )

    def forward(self, input):
        output = self.net(input)
        mu, logvar = split(output, self.embedding_dim, dim=1)

        return mu, logvar * self.max_log_var

    def reparameterize(self, input):
        mu, logvar = self(input)
        std = exp(logvar)
        eps = randn_like(std)
        output = self.emb_range_limit(mu + eps * std)
        return output

    def manual_forward(self, x, params):
        
        x = F.conv2d(x, params['conv1.weight'].to(device), params['conv1.bias'].to(device))
        dumy = torch.ones(np.prod(np.array(x.data.size()[1]))).cuda()*999999999999999999 # momentnum=1
        x = F.batch_norm(x, dumy, dumy, params['bn1.weight'], params['bn1.bias'], True, momentum=1)
        x = F.max_pool2d(F.relu(x), 2)
        
        x = F.conv2d(x, params['conv2.weight'].to(device), params['conv2.bias'].to(device))
        dumy = torch.ones(np.prod(np.array(x.data.size()[1]))).cuda()*999999999999999999 # momentnum=1
        x = F.batch_norm(x, dumy, dumy, params['bn2.weight'], params['bn2.bias'], True, momentum=1)
        x = F.max_pool2d(F.relu(x), 2)
        
        x = F.conv2d(x, params['conv3.weight'].to(device), params['conv3.bias'].to(device))
        dumy = torch.ones(np.prod(np.array(x.data.size()[1]))).cuda()*999999999999999999 # momentnum=1
        x = F.batch_norm(x, dumy, dumy, params['bn3.weight'], params['bn3.bias'], True, momentum=1)
        x = F.max_pool2d(F.relu(x), 2)
        
        x = x.view(x.size(0), self.h)
        x = F.linear(x, params['fc.weight'].to(device), params['fc.bias'].to(device))
        x = F.log_softmax(x, dim=1)
        
        return x


class MAML_Decoder(Module):
    def __init__(self, embedding_dim):
        super(MAML_Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.mlp = Sequential(
            Linear(self.embedding_dim, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            Linear(1024, 2048),
            BatchNorm1d(2048),
            LeakyReLU(),
            Linear(2048, 4096),
            BatchNorm1d(4096),
            LeakyReLU()
        )
        self.deconv = Sequential(
            ConvTranspose3d(64, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            ConvTranspose3d(64, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            ConvTranspose3d(64, 32, kernel_size=3, stride=3),
            BatchNorm3d(32),
            LeakyReLU(),
            ConvTranspose3d(32, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            ConvTranspose3d(16, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            ConvTranspose3d(8, 4, kernel_size=3, stride=1),
            BatchNorm3d(4),
            LeakyReLU(),
            ConvTranspose3d(4, 2, kernel_size=3, stride=1),
            BatchNorm3d(2),
            LeakyReLU(),
            ConvTranspose3d(2, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            LeakyReLU(),
            ConvTranspose3d(1, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            LeakyReLU(),
            ConvTranspose3d(1, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            LeakyReLU(),
            ConvTranspose3d(1, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            Tanh()
        )

    def forward(self, input):
        output = self.mlp(input)
        output = output.view(input.size()[0], 64, 4, 4, 4)
        return self.deconv(output)

class MetaLearner(object):
    def __init__(self):
        self.lr = 0.1
        self.momentum = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.master_net = OmniglotNet(n_class).to(self.device)
        self.master_opt = optim.Adam(self.master_net.parameters(), lr=0.001)
        self.keys = self.master_net.state_dict().keys()
    
    def copy_params(self, from_net, to_net):
        params = {k: v for k, v in from_net.state_dict().items() if k in self.keys}
        to_net.load_state_dict(params, strict=False)
    
    def save(self, model_path):
        torch.save(self.master_net.state_dict(), model_path)
    
    def load(self, model_path):
        self.master_net.load_state_dict(torch.load(model_path))
    
    def meta_test(self):
        
        meta_test_task_loader = TaskLoader(
            OmniglotOriginTaskset("../data/omniglot_mini/", meta_train=False, n_class=n_class, n_shot=n_shot))

        test_loss, test_acc = [], [] # For logging.
        
        sleep(0.5)
        with tqdm(meta_test_task_loader, desc="Meta Test ", ncols=10) as _tqdm:
            for meta_test_task in _tqdm:

                # copy master model to new branch model
                faster_net = OmniglotNet(n_class).to(self.device)
                self.copy_params(self.master_net, faster_net)
                faster_opt = optim.SGD(faster_net.parameters(), lr=self.lr, momentum=self.momentum)

                # make local task data loader
                local_task_train_data_loader = meta_test_task["train"]
                local_task_test_data_loader = meta_test_task["test"]

                # ----------------------------------------------------------------
                # meta test task train
                # ----------------------------------------------------------------

                for epoch in range(n_local_update):
                    _train_loss, _train_acc = train(
                        faster_net, self.device, local_task_train_data_loader, faster_opt, epoch)
                    _tqdm.set_postfix(OrderedDict(
                        epoch=epoch+1, 
                        train_loss="{:.3f}".format(_train_loss), 
                        train_acc="{:.3f}".format(_train_acc)))
                # ----------------------------------------------------------------
                # meta test task test
                # ----------------------------------------------------------------

                _test_loss, _test_acc = test(faster_net, self.device, local_task_test_data_loader)
                test_loss.append(_test_loss)
                test_acc.append(_test_acc)
        
        return np.mean(test_loss), np.mean(test_acc)

    
    def meta_train(self):
        
        meta_train_task_loader = TaskLoader(
            OmniglotAugmentedTaskset("../data/omniglot_mini/", meta_train=True, n_class=n_class, n_shot=n_shot))
    
        meta_grads = []
        
        test_loss, test_acc = [], [] # For logging.
        
        with tqdm(meta_train_task_loader, desc="Meta Train", ncols=10) as _tqdm:
            for meta_train_task in _tqdm:
                
                # copy master model to new branch model
                faster_net = OmniglotNet(n_class).to(self.device)
                faster_net.forward = NotImplementedError # goodbye!
                self.copy_params(self.master_net, faster_net)

                faster_params = OrderedDict((name, param) for (name, param) in faster_net.named_parameters())

                # make local task data loader
                local_task_train_data_loader = meta_train_task["train"]
                local_task_test_data_loader = meta_train_task["test"]

                # ----------------------------------------------------------------
                # meta train task train
                # ----------------------------------------------------------------

                first_train_for_this_task = True

                for epoch in range(n_local_update):
                    
                    _train_loss = 0 # For tqdm.
                    _train_acc = 0 # For tqdm.
                    
                    for data, target in local_task_train_data_loader:
                        data, target = data.to(self.device), target.to(self.device)

                        if first_train_for_this_task:
                            # manual predict
                            output = self.master_net(data)
                            loss = F.nll_loss(output, target)
                            pred = output.max(1, keepdim=True)[1]
                            
                            _train_loss += loss
                            _train_acc += pred.eq(target.view_as(pred)).sum().item()
                            
                            grads = torch.autograd.grad(loss, self.master_net.parameters(), create_graph=True)

                            first_train_for_this_task = False

                        else:
                            # manual predict
                            output = faster_net.manual_forward(data, faster_params)
                            loss = F.nll_loss(output, target)
                            pred = output.max(1, keepdim=True)[1]
                            
                            _train_loss += loss
                            _train_acc += pred.eq(target.view_as(pred)).sum().item()
                                                        
                            grads = torch.autograd.grad(loss, faster_params.values(), create_graph=True)
        
                        # manual optimize!!!
                        faster_params = OrderedDict(
                            (name, param - self.lr*grad)
                            for ((name, param), grad) in zip(faster_params.items(), grads)
                        )
                    
                    _train_loss /= len(local_task_train_data_loader.dataset)
                    _train_acc /= len(local_task_train_data_loader.dataset)
                    
                    _tqdm.set_postfix(OrderedDict(
                        epoch=epoch+1, 
                        train_loss="{:.3f}".format(_train_loss), 
                        train_acc="{:.3f}".format(_train_acc)))
                
                # ----------------------------------------------------------------
                # meta train task test
                # ----------------------------------------------------------------
                
                _test_loss = 0 # For logging.
                _test_acc = 0 # For logging.
                
                for data, target in local_task_test_data_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    output = faster_net.manual_forward(data, faster_params)
                    loss = F.nll_loss(output, target) # test_loss計算するとこまではfaster_net

                    # differentiates test_loss by master_net params
                    grads = torch.autograd.grad(loss, self.master_net.parameters(), retain_graph=True)
                    grads = {name:g for ((name, _), g) in zip(faster_net.named_parameters(), grads)}
                    meta_grads.append(grads)

                    pred = output.max(1, keepdim=True)[1]
                    acc = pred.eq(target.view_as(pred)).sum()
                    
                    _test_loss += loss.item()
                    _test_acc += acc.item()
                
                _test_loss /= len(local_task_test_data_loader.dataset)
                _test_acc /= len(local_task_test_data_loader.dataset)  
                test_loss.append(_test_loss)
                test_acc.append(_test_acc)
        
        # ----------------------------------------------------------------
        # end all tasks
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        # meta update
        # ----------------------------------------------------------------
        
        meta_grads = {k: sum(grads[k] for grads in meta_grads) for k in meta_grads[0].keys()}
        
        # using data,target from somewhere
        dumy_output = self.master_net(data)
        dumy_loss = F.nll_loss(dumy_output, target)
        
        # after dumy_loss.backward, rewrite grads
        self.master_opt.zero_grad()
        dumy_loss.backward(retain_graph=True)

        hooks = []
        for (k,v) in self.master_net.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return meta_grads[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))

        # Compute grads for current step, replace with summed gradients as defined by hook
        self.master_opt.zero_grad()
        dumy_loss.backward()

        # Update the net parameters with the accumulated gradient according to optimizer
        self.master_opt.step()

        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

        return np.mean(test_loss), np.mean(test_acc)

if __name__=="__main__":
    embedding_dim = 686
    encoder = MAML_Encoder(686)
    decoder = MAML_Decoder(686)
    print(encoder)
    print(encoder.state_dict().keys())