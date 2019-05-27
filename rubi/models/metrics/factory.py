from bootstrap.lib.options import Options
from block.models.metrics.vqa_accuracies import VQAAccuracies
from .vqa_rubi_metrics import VQARUBiMetrics

def factory(engine, mode):
    name = Options()['model.metric.name']
    metric = None

    if name == 'vqa_accuracies':
        open_ended = ('tdiuc' not in Options()['dataset.name'] and 'gqa' not in Options()['dataset.name'])
        if mode == 'train':
            split = engine.dataset['train'].split
            if split == 'train':
                metric = VQAAccuracies(engine,
                    mode='train',
                    open_ended=open_ended,
                    tdiuc=True,
                    dir_exp=Options()['exp.dir'],
                    dir_vqa=Options()['dataset.dir'])
            elif split == 'trainval':
                metric = None
            else:
                raise ValueError(split)
        elif mode == 'eval':
            metric = VQAAccuracies(engine,
                mode='eval',
                open_ended=open_ended,
                tdiuc=('tdiuc' in Options()['dataset.name'] or Options()['dataset.eval_split'] != 'test'),
                dir_exp=Options()['exp.dir'],
                dir_vqa=Options()['dataset.dir'])
        else:
            metric = None

    elif name == "vqa_rubi_metrics":
        open_ended = ('tdiuc' not in Options()['dataset.name'] and 'gqa' not in Options()['dataset.name'])
        metric = VQARUBiMetrics(engine,
            mode=mode,
            open_ended=open_ended,
            tdiuc=True,
            dir_exp=Options()['exp.dir'],
            dir_vqa=Options()['dataset.dir']
        )

    else:
        raise ValueError(name)
    return metric
