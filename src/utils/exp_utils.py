import os

def create_exp_dir(exp_name):
    os.makedirs('exps', exist_ok=True)
    exp_dir = os.path.join('exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    sub_dirs = ['checkpoints', 'configs', 'data', 'results']

    for dir in sub_dirs:
        dir_path = os.path.join(exp_dir, dir)
        os.makedirs(dir_path, exist_ok=True)

    results_dir = os.path.join(exp_dir, 'results')
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    data_dir = os.path.join(exp_dir, 'data')
    configs_dir = os.path.join(exp_dir, 'configs')

    return (
        exp_dir, 
        configs_dir,
        data_dir, 
        checkpoints_dir,
        results_dir
    )



def summarize_results():
    pass
