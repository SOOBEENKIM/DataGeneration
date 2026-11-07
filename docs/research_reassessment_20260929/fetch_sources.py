"""Read-only primary-literature snapshot; no datasets or model execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib,json
from html.parser import HTMLParser
from pathlib import Path
import requests

OUT=Path(__file__).resolve().parent/'literature'
PAPERS={'argn':'2501.12012','tabdit':'2504.07566','seq2synth':'2607.15606',
 'tabstruct':'2509.11950','tabcascade':'2601.22816','emdt':'2603.13566',
 'lbdtpp':'2606.24982','motpp':'2609.21382','sigtpp':'2607.06652',
 'omegasds':'2605.06315','adaptive_seqdiff':'2606.06007','lin':'2604.27182',
 'fllm':'2602.12756','behavioral_fraud':'2604.13125','timeautodiff':'2406.16028',
 'marked_lnm':'2210.15294','conditional_w1':'2403.18705','ctdf':'2606.28674',
 'ifltpp':'1909.12127','cpar_finance':'2401.00965','cpar':'2207.14406','adiff4tpp':'2504.20411'}

class Text(HTMLParser):
    def __init__(self):super().__init__();self.out=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ['script','style']:self.skip+=1
        if tag in ['p','h1','h2','h3','h4','h5','h6','tr','li','section','div']:self.out.append('\n')
    def handle_endtag(self,tag):
        if tag in ['script','style']:self.skip=max(0,self.skip-1)
        if tag in ['p','h1','h2','h3','h4','h5','h6','tr','li','section']:self.out.append('\n')
    def handle_data(self,data):
        if not self.skip:self.out.append(data)

def fetch(item):
    key,identifier=item;url='https://arxiv.org/html/'+identifier
    try:
        response=requests.get(url,timeout=50);response.raise_for_status()
        data=response.content;(OUT/f'{key}.html').write_bytes(data)
        parser=Text();parser.feed(response.text)
        lines=[' '.join(x.split()) for x in ''.join(parser.out).splitlines()]
        text='\n'.join(x for x in lines if x)
        (OUT/f'{key}.txt').write_text(text)
        record=dict(key=key,url=url,resolved=response.url,status=response.status_code,sha256=hashlib.sha256(data).hexdigest(),characters=len(text))
    except Exception as error:record=dict(key=key,url=url,error=repr(error))
    print(json.dumps(record),flush=True);return record

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,PAPERS.items()))
    (OUT/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),sources=records),indent=2)+'\n')
