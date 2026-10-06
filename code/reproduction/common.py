import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from . import paths


def read_data(name):
    return json.loads((paths.ROOT / 'data' / name).read_text(encoding='utf-8'))


def save_json(name, value):
    (paths.OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite paired observations are required')
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {'mean':mean, 'ci95':[mean-half,mean+half], 'n':len(x)}


def assert_numeric_match(actual, expected, path='root'):
    """Portable numeric audit; do not require byte-identical floating point."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_numeric_match(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a,b) in enumerate(zip(actual,expected)):
            assert_numeric_match(a,b,path+'/'+str(i))
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10,err_msg=path)
    else:
        assert actual == expected, (path, actual, expected)


def save_figure(fig, name):
    for suffix in ['pdf','svg','png']:
        kwargs = {'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
        fig.savefig(paths.OUT / f'{name}.{suffix}', dpi=240, **kwargs)
