# top-import for cuda device initialization
from cuda import set_cuda

import logger

from cli import parse_command
from losses import get_loss
from utils import get_xp, set_seed
from data import get_data_loaders
from models import get_model, load_best_model
from optim import get_optimizer, decay_optimizer
from epoch import train, test

def move_set_to_cuda(loader, args):
    res = []
    for x, y in loader:
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)
        res.append((x,y))
    return res

def main(args):

    set_cuda(args)
    set_seed(args)

    loader_train, loader_val, loader_test = get_data_loaders(args)
    loss = get_loss(args)
    model = get_model(args)
    optimizer = get_optimizer(args, parameters=model.parameters())
    xp = get_xp(args, model, optimizer)

    print("Maximal number of epochs:\t{}\n".format(args.epochs))

    train_loader_in_cuda = move_set_to_cuda(loader_train, args)
    # move to cuda only once

    for i in range(args.epochs):
        xp.Epoch.update(1).log()



        train(model, loss, optimizer, train_loader_in_cuda, xp, args)
        test(model, loader_val, xp, args)

        if (i + 1) in args.T:
            decay_optimizer(optimizer, args.decay_factor)

    load_best_model(model, xp)
    test(model, loader_test, xp, args)


if __name__ == '__main__':
    args = parse_command()
    with logger.stdout_to("{}/log.txt".format(args.xp_name)):
        main(args)
