from torch.optim.optimizer import Optimizer, required
import torch
from torch.autograd import Variable
import copy, logging


class SVRG(torch.optim.SGD):
    """Implements stochastic variance reduction gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        T (int): number of iterations between the step to take the full grad/save w
        data_loader (DataLoader): dataloader to use to load training data
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum (default: 0)
        opt (torch.optim): optimizer to baseclass (default: SGD)
    """

    def __init__(self, params, lr=required, T=required, data_loader=required, weight_decay=0.0,
                 momentum=0.0, opt=torch.optim.SGD):

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Choose the baseclass dynamically.
        self.__class__ = type(self.__class__.__name__,
                              (opt,object),
                              dict(self.__class__.__dict__))
        logging.info("Using base optimizer {} in SVRG".format(opt))
        super(self.__class__, self).__init__(params, **defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SVRG doesn't support per-parameter options "
                             "(parameter groups)")

        params = self.param_groups[0]['params']

        self._params = params

        self._curr_w = [p.data for p in params]
        self._prev_w = [p.data.clone() for p in params]

        # Gradients are lazily allocated and don't exist yet. However, gradients are
        # the same shape as the weights so we can still allocate buffers here
        self._curr_grad = [p.data.clone() for p in params]
        self._prev_grad = [p.data.clone() for p in params]
        self._full_grad = None

        self.data_loader = data_loader
        self.state['t_iters'] = T
        self.T = T # Needed to trigger full gradient

        logging.info("Data Loader has {} with batch {}".format(len(self.data_loader),
                                                               self.data_loader.batch_size))

    def __setstate__(self, state):
        super(self.__class__, self).__setstate__(state)

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()

    def _set_weights_grad(self,ws,gs):
        for idx, p in enumerate(self._params):
            if ws is not None: p.data = ws[idx]
            if gs is not None and p.grad is not None: p.grad.data = gs[idx]
            if (gs is not None) and (p.grad is not None):
                assert (p.grad.data.data_ptr() == gs[idx].data_ptr())

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Calculate full gradient
        if self.state['t_iters'] == self.T:
            # Setup the full grad
            # Reset gradients before accumulating them
            self._set_weights_grad(None, self._full_grad)
            self._zero_grad()

            # Accumulate gradients
            for i, (data, target) in enumerate(self.data_loader):
                closure(data, target)

            # Adjust summed gradients by num_iterations accumulated over
            # assert(n_iterations == len(self.data_loader))
            for p in self._params:
                if p.grad is not None:
                    p.grad.data /= len(self.data_loader)

            if self._full_grad is None:
                self._full_grad = []
                for p in self._params:
                    if p.grad is not None:
                        self._full_grad.append(p.grad.data.clone())
                    else:
                        self._full_grad.append(None)

            # Copy w to prev_w
            for p, p0 in zip(self._curr_w, self._prev_w):
                p0.copy_(p)

            # Reset t
            self.state['t_iters'] = 0

        # Setup the previous grad
        self._set_weights_grad(self._prev_w, self._prev_grad)
        self._zero_grad()
        closure()

        # Calculate the current grad.
        self._set_weights_grad(self._curr_w, self._curr_grad)
        self._zero_grad()
        loss = closure()

        # Adjust the current gradient using the previous gradient and the full gradient.
        # We have normalized so that these are all comparable.
        for p, d_p0, fg in zip(self._params, self._prev_grad, self._full_grad):
            # Adjust gradient in place
            if p.grad is not None:
                p.grad.data -= (d_p0 - fg)

        # Call optimizer update step
        super(self.__class__, self).step()

        self.state['t_iters'] += 1
        return loss