"""Full-size timing probe on random latents; no data fitting or saved weights."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/meld_600_pilot_v1'
sys.path.insert(0, str(ROOT / 'artifacts/comparison_completion_v1/local_source'))
import meld_train as m

def main():
    m.torch.set_num_threads(2)
    m.verify()
    config = json.loads((m.DATA / 'VAE_CONFIG.json').read_text())
    model, _ = m.make_dit(config['seq_len'])
    transport = m.create_transport('Linear', 'velocity', use_cosine_loss=True, use_lognorm=True)
    x = m.torch.randn(8, 32, config['seq_len'], device='cuda')
    text = ['gender: F; state: CA; birth_year: 1980; city_pop: 10000'] * 8
    times = []
    m.torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        model.zero_grad(set_to_none=True)
        m.torch.cuda.synchronize(); start = time.monotonic()
        for _ in range(4):
            loss = m.losses(transport, model, x, text) / 4
            assert m.torch.isfinite(loss)
            loss.backward()
        m.torch.cuda.synchronize(); times.append(time.monotonic() - start)
        print('PROBE', i, times[-1], flush=True)
    result = dict(seconds_per_batch32=times, steady_mean=sum(times[1:])/3,
                  peak_gib=m.torch.cuda.max_memory_allocated()/2**30,
                  gpu=os.environ['CUDA_VISIBLE_DEVICES'], random_latents=True,
                  optimizer_steps=0, performance_training=False,
                  note='Excludes optimizer, EMA, validation and generation; measured on idle GPU.')
    m.write(OUT / 'BUDGET_PROBE.json', result)

if __name__ == '__main__':
    main()
