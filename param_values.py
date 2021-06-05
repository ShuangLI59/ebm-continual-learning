
def set_default_values(args, also_hyper_params=True):
    # -set default-values for certain arguments based on chosen scenario & experiment
    if args.tasks is None:
        if args.experiment=='splitMNIST':
            args.num_classes = 10
        if args.experiment=='splitMNISToneclass':
            args.num_classes = 10
        elif args.experiment=='permMNIST':
            args.num_classes = 100
        elif args.experiment=='cifar10':
            args.num_classes = 10
        elif args.experiment=='cifar100':
            args.num_classes = 100

    if args.iters is None:
        if args.experiment=='splitMNIST':
            args.iters = 2000
        elif args.experiment=='splitMNISToneclass':
            args.iters = 2000
        elif args.experiment=='permMNIST':
            args.iters = 5000
        elif args.experiment=='cifar100':
            args.iters = 5000
        elif args.experiment=='cifar10':
            args.iters = 5000
        elif args.experiment=='block2d':
            args.iters = 5000
    
    if args.lr is None:
        if args.ebm:
            if args.experiment=='splitMNIST':
                args.lr = 0.0001
            if args.experiment=='splitMNISToneclass':
                args.lr = 0.0001
            elif args.experiment=='permMNIST':
                args.lr = 0.00001
            elif args.experiment=='cifar100':
                args.lr = 0.00001
            elif args.experiment=='cifar10':
                args.lr = 0.00001
            elif args.experiment=='block2d':
                args.lr = 0.00001
        else:
            if args.experiment=='splitMNIST':
                args.lr = 0.001
            if args.experiment=='splitMNISToneclass':
                args.lr = 0.001
            elif args.experiment=='permMNIST':
                args.lr = 0.0001
            elif args.experiment=='cifar100':
                args.lr = 0.0001
            elif args.experiment=='cifar10':
                args.lr = 0.0001
            elif args.experiment=='block2d':
                args.lr = 0.0001


    if args.fc_units is None:
        if args.experiment=='splitMNIST':
            args.fc_units = 400
        if args.experiment=='splitMNISToneclass':
            args.fc_units = 400
        elif args.experiment=='permMNIST':
            args.fc_units = 1000
        elif args.experiment=='cifar100':
            args.fc_units = 1000
        elif args.experiment=='cifar10':
            args.fc_units = 1000
        elif args.experiment=='block2d':
            args.fc_units = 400


    print(args)
    return args



