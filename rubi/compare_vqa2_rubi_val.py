import argparse
from bootstrap.compare import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--nb_epochs', default=-1, type=int)
    parser.add_argument('-d', '--dir_logs', default='', type=str, nargs='*')
    parser.add_argument('-m', '--metrics', type=str, action='append', nargs=3,
                        metavar=('json', 'name', 'order'),
                        default=[['logs', 'eval_epoch.accuracy_top1', 'max'],
                                ['logs', 'eval_epoch.loss', 'min'], 
                                ['logs', 'eval_epoch.loss', 'min'], 
                                ['logs', 'eval_epoch.loss', 'min'], 
                                ['logs', 'eval_epoch.loss', 'min'], 
                                ['logs', 'eval_epoch.loss', 'min'], 
                                ['logs', 'eval_epoch.loss', 'min'], 
                                ['logs_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_q_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_v_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_mm_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_mm_v_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_mm_q_val_oe', 'eval_epoch.overall', 'max'],
                                ['logs_mm_v_q_val_oe', 'eval_epoch.overall', 'max'],
                                ])
    parser.add_argument('-b', '--best', type=str, nargs=3,
                        metavar=('json', 'name', 'order'),
                        default=None)
    args = parser.parse_args()
    main(args)
