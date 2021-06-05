
def get_param_stamp(args, model_name, verbose=True):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}".format(n=args.tasks, set=args.scenario) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(exp=args.experiment, multi_n=multi_n_stamp)
    if verbose:
        print("\n"+" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}-{optim}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr==args.lr_gen else "-lrG{}".format(args.lr_gen)) if hasattr(args, "lr_gen") else "",
        bsz=args.batch, optim=args.optimizer,
    )

    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)


    # --> combine
    param_stamp = "{}--{}--{}{}".format(
        task_stamp, model_stamp, hyper_stamp, "-s{}".format(args.seed) if not args.seed==0 else "")

    ## Print param-stamp on screen and return
    print(param_stamp)
    return param_stamp