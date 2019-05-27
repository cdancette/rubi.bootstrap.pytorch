from bootstrap.lib.options import Options
from block.models.criterions.vqa_cross_entropy import VQACrossEntropyLoss
from .rubi_criterion import RUBiCriterion

def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split
    eval_only = 'train' not in engine.dataset
    
    opt = Options()['model.criterion']
    if split == "test" and 'tdiuc' not in Options()['dataset.name']:
        return None
    if name == 'vqa_cross_entropy':
        criterion = VQACrossEntropyLoss()
    elif name == "rubi_criterion":
        criterion = RUBiCriterion(
            question_loss_weight=opt['question_loss_weight']
        )
    else:
        raise ValueError(name)
    return criterion
