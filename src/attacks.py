import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
none_std = (1.0, 1.0, 1.0)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
none_std = torch.tensor(none_std).view(3, 1, 1).cuda()


def loss_kl(x_adv, x_natural, y, model):
    criterion_kl = nn.KLDivLoss(size_average=False)
    return criterion_kl(F.log_softmax(model(x_adv), dim=1),
                        F.softmax(model(x_natural), dim=1))


def privacy_loss(x_adv, x_natural, y, model1, model2, target1, target2):
    preds1 = model1(x_adv)
    preds2 = model2(x_adv)
    p = preds1
    q = preds2
    p_soft_log = F.log_softmax(p)
    q_soft = F.softmax(q)
    kl_loss = torch.nn.KLDivLoss(size_average=False)(p_soft_log, q_soft)
    return kl_loss


def patch_attack_private(model1, model2, X, y, params, target1, target2):
    # ipdb.set_trace()
    epsilon = params.epsilon_l_1
    alpha = params.alpha_l_1
    num_iter = params.num_iter
    model1.eval()
    model2.eval()
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = privacy_loss(X + delta, X, y, model1, model2, target1, target2)
        loss.backward()
        delta.data = (delta.data + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        delta.grad.zero_()

    print(y)
    pred1 = model1(X + delta).max(1)[1]
    print(pred1)
    pred2 = model2(X + delta).max(1)[1]
    print(pred2)
    print("Accuracy: ", (pred1 != pred2).sum().item())
    return delta


def loss_crossentropy(x_adv, x_natural, y, model, target=None):
    preds = model(x_adv)
    if target is None:
        loss = nn.CrossEntropyLoss()(preds, y)
    else:
        loss = -1 * nn.CrossEntropyLoss()(preds, target)
    return loss


def loss_mingd(preds, target):
    loss = (preds.max(dim=1)[0] - preds[torch.arange(preds.shape[0]), target]).mean()
    assert (loss >= 0)
    return loss


def rand_steps(model, X, y, args, target=None):
    # optimized implementation to only query remaining points
    del target  # The attack does not use the targets
    start = time.time()
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts

    # Define the Noise
    uni, std, scale = (0.005, 0.005, 0.01)
    steps = 50
    noise_2 = lambda X: torch.normal(0, std, size=X.shape).cuda()
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(X.device)
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni, uni)

    noise_map = {"l1": noise_1, "l2": noise_2, "linf": noise_inf}
    mag = 1

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)
    with torch.no_grad():
        for t in range(steps):
            if t > 0:
                preds = model(X_r + delta_r)
                new_remaining = (preds.max(1)[1] == y[remaining])
                remaining[remaining] = new_remaining
            else:
                preds = model(X + delta)
                remaining = (preds.max(1)[1] == y)

            if remaining.sum() == 0:
                break

            mag += 1
            X_r = X[remaining]
            delta_r = delta_base[remaining] * mag
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1 - X_r)  # clip X+delta_r[remaining] to [0,1]
            delta[remaining] = delta_r.detach()

        print(f"Number of steps = {t + 1} | Failed to convert = {(model(X + delta).max(1)[1] == y).sum().item()} "
              f"| Time taken = {time.time() - start}")
    if is_training:
        model.train()
    return delta


def mingd(model, X, y, args, target):
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts
    alpha_map = {"l1": args.alpha_l_1 / args.k, "l2": args.alpha_l_2, "linf": args.alpha_l_inf}
    alpha = float(alpha_map[args.distance])

    delta = torch.zeros_like(X, requires_grad=False)
    preds = model(X + delta)
    remaining = (preds.max(1)[1] != target)
    X_r = X[remaining]
    delta_r = delta[remaining]
    delta_r.requires_grad = True
    if remaining.sum() == 0:
        if is_training:
            model.train()
        return delta

    for t in range(args.num_iter):
        # Pass inputs as parameters
        preds = model(X_r + delta_r)

        # Calculate loss and gradient of inputs
        loss = -1 * loss_mingd(preds, target[remaining])
        loss.backward()
        grads = delta_r.grad.detach()

        # Calculate which inputs have not reached correct predictions
        new_remaining = (preds.max(1)[1] != target[remaining])
        remaining[remaining.clone()] = new_remaining
        if remaining.sum() == 0:
            break

        # Collect remaining values
        X_r = X[remaining]
        delta_r = delta[remaining]
        delta_r.requires_grad = True
        grads = grads[new_remaining]

        # Calculate new offset from gradient
        if args.distance == "linf":
            delta_r.data += alpha * grads.sign()
        elif args.distance == "l2":
            delta_r.data += alpha * (grads / norms_l2(grads + 1e-12))
        elif args.distance == "l1":
            delta_r.data += alpha * l1_dir_topk(grads, delta_r.data, X_r, args.gap, args.k)
        delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1 - X_r)  # clip X+delta_r[remaining] to [0,1]
        delta[remaining] = delta_r.detach()

    # print(f"Number of steps = {t + 1} | Failed to convert = {(model(X + delta).max(1)[1] != target).sum().item()} "
    #       f"| Time taken = {time.time() - start}")
    if is_training:
        model.train()
    return delta


def pgd_linf(model, X, y, params, target=None):
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts
    params = vars(params) if type(params) != type({"a": 1}) else params
    epsilon = params.get('epsilon_l_inf')
    alpha = params.get('alpha_l_inf')
    num_iter = params.get('num_iter')
    restarts = params.get('restarts')
    randomize = params.get('randomize')
    smallest_adv = params.get('smallest_adv')
    criterion = loss_kl if params.get('criterion') == "TRADES" else loss_crossentropy
    normalize = 0 if not params.get('normalize') else 1
    if normalize and params.get('dataset') == "CIFAR10":
        upper_limit = ((1 - mu) / std)
        lower_limit = ((0 - mu) / std)
        epsilon = epsilon / std
        alpha = alpha / std
    else:
        lower_limit = 0
        upper_limit = 1
        epsilon = epsilon / none_std
        alpha = alpha / none_std

    if randomize == 2:
        randomize = np.random.randint(2) if restarts == 1 else 1
        # If there are more than 1 restarts, anyways the following loop ensures that atleast one of the starts is from 0 when rand = 1

    assert (restarts >= 1)
    if alpha == None:
        alpha = epsilon * 0.01 / 0.3

    max_delta = torch.zeros_like(X, requires_grad=False).cpu()

    for i in range(restarts):
        delta = torch.empty_like(X).uniform_(-1, 1) * epsilon.unsqueeze(-1).unsqueeze(-1)
        delta.requires_grad = True
        if i == 0 and (randomize == 0 or restarts > 1):
            # Make a 0 initialization if you are making multiple restarts
            # or if explicitly told not to randomize for a single start
            delta = torch.zeros_like(X, requires_grad=True)
        for t in range(num_iter):
            if smallest_adv:
                output = model(X + delta)
                if target is not None:
                    correct = output.max(1)[1] != target
                else:
                    correct = (output.max(1)[1] == y)
                correct = correct.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            else:
                correct = 1.
            # Finding the correct examples so as to attack only them
            loss = criterion(X + delta, X, y, model, target)
            loss.backward()
            delta.data = torch.min(torch.max((delta.data + alpha * correct * delta.grad.detach().sign()), -epsilon),
                                   epsilon)
            delta.data = torch.min(torch.max(delta.detach(), lower_limit - X), upper_limit - X)  # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X + delta)
        incorrect = output.max(1)[1] != y
        # Edit Max Delta only for successful attacks
        if i == 0:
            max_delta = delta.detach().cpu()
        else:
            max_delta[incorrect] = delta.detach()[incorrect].cpu()

        del delta, loss, output, incorrect
        torch.cuda.empty_cache()

    if is_training:
        model.train()  # Reset to train mode if model was training earlier
    return max_delta.to(params['device'])


def pgd_l2(model, X, y, params, target=None):
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts
    params = vars(params) if type(params) != type({"a": 1}) else params
    epsilon = params.get('epsilon_l_2')
    alpha = params.get('alpha_l_2')
    num_iter = params.get('num_iter')
    restarts = params.get('restarts')
    randomize = params.get('randomize')
    smallest_adv = params.get('smallest_adv')
    criterion = loss_kl if params.get('criterion') == "TRADES" else loss_crossentropy
    normalize = 0 if not params.get('normalize') else 1
    if normalize and params.get('dataset') == "CIFAR10":
        upper_limit = ((1 - mu) / std)
        lower_limit = ((0 - mu) / std)
        epsilon = epsilon / std
        alpha = alpha / std
    else:
        lower_limit = 0
        upper_limit = 1
        epsilon = epsilon / none_std
        alpha = alpha / none_std

    if randomize == 2:
        randomize = np.random.randint(2) if restarts == 1 else 1
        # If there are more than 1 restarts, anyways the following loop ensures that atleast one of the starts is from 0 when rand = 1

    assert (restarts >= 1)
    max_delta = torch.zeros_like(X, requires_grad=False).cpu()

    for i in range(restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data *= (2.0 * delta.data - 1.0)
        delta.data = delta.data * epsilon / norms_l2(delta.detach())
        if i == 0 and (randomize == 0 or restarts > 1):
            # Make a 0 initialization if you are making multiple restarts
            # or if explicitly told not to randomize for a single start
            delta = torch.zeros_like(X, requires_grad=True)

        for t in range(num_iter):
            if smallest_adv:
                output = model(X + delta)
                if target is not None:
                    correct = output.max(1)[1] != target
                else:
                    correct = (output.max(1)[1] == y)
                correct = correct.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            else:
                correct = 1.
            # Finding the correct examples so as to attack only them
            loss = criterion(X + delta, X, y, model, target)
            loss.backward()
            delta.data += correct * alpha * delta.grad.detach() / norms_l2(delta.grad.detach())
            delta.data *= epsilon / torch.max(norms_l2(delta.detach()), epsilon)
            delta.data = torch.min(torch.max(delta.detach(), lower_limit - X), upper_limit - X)  # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X + delta)
        incorrect = output.max(1)[1] != y
        # Edit Max Delta only for successful attacks
        if i == 0:
            max_delta = delta.detach().cpu()
        else:
            max_delta[incorrect] = delta.detach()[incorrect].cpu()

        del delta, loss, output, incorrect
        torch.cuda.empty_cache()
    if is_training:
        model.train()  # Reset to train mode if model was training earlier
    return max_delta.to(params['device'])


def pgd_l1(model, X, y, params, target=None):
    is_training = model.training
    model.eval()  # Need to freeze the batch norm and dropouts
    params = vars(params) if type(params) != type({"a": 1}) else params
    epsilon = params.get('epsilon_l_1')
    alpha = params.get('alpha_l_1')
    num_iter = params.get('num_iter')
    restarts = params.get('restarts')
    randomize = params.get('randomize')
    smallest_adv = params.get('smallest_adv')
    device = params.get('device')
    gap = params.get('gap')
    k = params.get('k')
    criterion = loss_kl if params.get('criterion') == "TRADES" else loss_crossentropy
    normalize = 0 if not params.get('normalize') else 1
    if normalize and params.get('dataset') == "CIFAR10":
        upper_limit = ((1 - mu) / std)
        lower_limit = ((0 - mu) / std)
        epsilon = epsilon / std
        alpha = alpha / std
    else:
        lower_limit = 0
        upper_limit = 1
        # epsilon = epsilon / none_std
        # alpha = alpha / none_std

    if randomize == 2:
        randomize = np.random.randint(2) if restarts == 1 else 1
        # If there are more than 1 restarts, anyways the following loop ensures that atleast one of the starts is from 0 when rand = 1

    assert (restarts >= 1)
    # Gap : Dont attack pixels closer than the gap value to 0 or 1

    max_delta = torch.zeros_like(X, requires_grad=False).cpu()
    alpha = alpha / float(k)

    for i in range(restarts):
        delta = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(device)
        delta.data = delta.data * epsilon / norms_l1(delta.detach())
        delta.requires_grad = True
        if i == 0 and (randomize == 0 or restarts > 1):
            # Make a 0 initialization if you are making multiple restarts
            # or if explicitly told not to randomize for a single start
            delta = torch.zeros_like(X, requires_grad=True)

        for t in range(num_iter):
            if smallest_adv:
                output = model(X + delta)
                if target is not None:
                    correct = output.max(1)[1] != target
                else:
                    correct = (output.max(1)[1] == y)
                correct = correct.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            else:
                correct = 1.
            # Finding the correct examples so as to attack only them
            loss = criterion(X + delta, X, y, model, target)
            loss.backward()
            delta.data += alpha * correct * l1_dir_topk(delta.grad.detach(), delta.data, X, gap, k)
            if (norms_l1(delta) > epsilon).any() and not normalize:  ## Does not support normalized values as of now
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), lower_limit - X), upper_limit - X)  # clip X+delta to [0,1]
            delta.grad.zero_()
        output = model(X + delta)
        incorrect = output.max(1)[1] != y
        # Edit Max Delta only for successful attacks
        if i == 0:
            max_delta = delta.detach().cpu()
        else:
            max_delta[incorrect] = delta.detach()[incorrect].cpu()

        del delta, loss, output, incorrect
        torch.cuda.empty_cache()
    if is_training:
        model.train()  # Reset to train mode if model was training earlier
    return max_delta.to(params['device'])


def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim=dim)
    return val[:, :, -1], idx[:, :, -1]


def l1_dir_topk(grad, delta, X, gap, k=10):
    # Check which all directions can still be increased such that
    # they haven't been clipped already and have scope of increasing
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    neg1 = (grad < 0) * (X_curr <= gap)
    neg2 = (grad > 0) * (X_curr >= 1 - gap)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size, 1, -1)
    grad_check = grad.view(batch_size, 1, -1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim=2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=10, device="cuda:1"):
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim=(1, 2, 3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
        # check if x is already a solution
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device=device)
    # compute the solution to the original problem on v
    y *= x.sign()
    y *= epsilon / norms_l1(y)
    return y


def proj_simplex(v, s=1, device="cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get 'the' array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size, 1, -1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending=True)
    cssv = u.cumsum(dim=2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n + 1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim=2)
    w = (comp - 1).cumsum(dim=2)
    u = u + w
    rho = torch.argmax(u, dim=2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i, 0, rho[i]] for i in range(cssv.shape[0])]).to(device)
    c = c - s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c, (rho.float() + 1))
    theta = theta.view(batch_size, 1, 1, 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def norms_l2(Z):
    return norms(Z)


def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)


def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]


def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None].squeeze(1).squeeze(1).squeeze(1)


def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1) != 0).sum(dim=1)[:, None, None, None]).float()


def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1) != 0).sum(dim=1)[:, None, None, None]).float().squeeze(1).squeeze(1).squeeze(1)


def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]
