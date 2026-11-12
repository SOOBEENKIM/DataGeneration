"""Small CPU controls followed by matched native generations on idle GPUs."""
from unittest.mock import patch
from study import OUT,DOCS,SEEDS,PARENTS,NEW_ARMS,verify
import pipeline as dispatcher
def jobs():
    result=[]
    for arm in NEW_ARMS:
        for rs in SEEDS:
            for parent in PARENTS:
                args=['--seed',str(rs),'--parent',str(parent),'--arm',arm]
                fitted=OUT/f'fits/{arm}_{parent}_{rs}/COMPLETE.json'
                result.append(dict(key=f'fit_{arm}_{parent}_{rs}',mode='fit',kind='cpu',
                    deps=[str(OUT/'READY.json')],done=str(fitted),args=args))
                result.append(dict(key=f'gen_{arm}_{parent}_{rs}',mode='generate',kind='gpu',
                    deps=[str(fitted),str(OUT/'EVALUATION_READY.json')],done=str(OUT/f'runs/{arm}_{rs}_{parent}/COMPLETE.json'),args=args))
    return result
if __name__=='__main__':
    with patch.object(dispatcher,'OUT',OUT),patch.object(dispatcher,'DOCS',DOCS), \
         patch.object(dispatcher,'SCRIPT',OUT/'local_source/study.py'),patch.object(dispatcher,'verify',verify),patch.object(dispatcher,'jobs',jobs):
        dispatcher.main()
