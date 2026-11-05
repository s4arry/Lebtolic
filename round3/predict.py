"""Predict with a trusted round3 model.json. No executable pickle files needed."""
import argparse
import json
from pathlib import Path

import numpy as np

from .workflow import load_original, predict_demo


class PortableModel:
    def __init__(self, state):
        self.state = state

    def predict_proba(self, rows):
        s = self.state
        x = np.asarray(rows, float)
        x = np.where(np.isnan(x), s['imputer_median'], x)
        x = (x-s['scaler_mean'])/s['scaler_scale']
        if s['selected']['family'] == 'logistic':
            z = x @ np.array(s['coefficients'])[0] + s['intercept'][0]
            p = np.exp(-np.logaddexp(0., -z))
        else:
            x = x.astype(np.float32)  # sklearn tree traversal converts inputs to float32.
            tree = s['tree']; probabilities = []
            for row in x:
                node = 0
                for _ in range(len(tree['feature'])):
                    if tree['children_left'][node] == -1:
                        value = np.array(tree['value'][node][0], float)
                        probabilities.append(value[1]/value.sum()); break
                    key = 'children_left' if row[tree['feature'][node]] <= tree['threshold'][node] else 'children_right'
                    node = tree[key][node]
                else:
                    raise ValueError('Invalid tree structure')
            p = np.array(probabilities)
        return np.column_stack([1-p, p])


def predict_file(path, values, previously_diagnosed=False):
    state = json.loads(Path(path).read_text(encoding='utf-8'))
    return predict_demo(PortableModel(state), state['calibration'], load_original(),
                        state['training_range'], state['level_cuts'], values, previously_diagnosed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='JSON object with baseline features')
    parser.add_argument('--previously-diagnosed', action='store_true')
    args = parser.parse_args()
    print(json.dumps(predict_file(args.model, json.loads(args.input.read_text(encoding='utf-8')),
                                  args.previously_diagnosed), ensure_ascii=False, indent=2))
