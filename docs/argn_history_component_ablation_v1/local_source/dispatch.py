"""Reuse the proven idle-GPU dispatcher with a new fixed generation-only queue."""
from unittest.mock import patch
from study import OUT,DOCS,SEEDS,PARENTS,NEW_ARMS,verify
import pipeline as dispatcher

def jobs():
    return [dict(key=f'gen_{arm}_{parent}_{rs}',mode='generate',kind='gpu',
        deps=[str(OUT/'READY.json')],done=str(OUT/f'runs/{arm}_{rs}_{parent}/COMPLETE.json'),
        args=['--seed',str(rs),'--parent',str(parent),'--arm',arm])
        for rs in SEEDS for parent in PARENTS for arm in NEW_ARMS]

if __name__=='__main__':
    with patch.object(dispatcher,'OUT',OUT),patch.object(dispatcher,'DOCS',DOCS), \
         patch.object(dispatcher,'SCRIPT',OUT/'local_source/study.py'), \
         patch.object(dispatcher,'verify',verify),patch.object(dispatcher,'jobs',jobs):
        dispatcher.main()
