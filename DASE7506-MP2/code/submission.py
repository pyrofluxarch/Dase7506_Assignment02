"""MP2 submission: linear candidate ranker with numeric-consistency features.

Inference dependencies: NumPy only (plus the assignment's common.py parser).
The checkpoint directory should contain model.npz.
"""
from collections import Counter
from fractions import Fraction
import math
import re
from pathlib import Path
import numpy as np

try:
    from common import parse_answer
except ImportError:  # fallback for standalone inspection outside the course repo
    import re as _re
    def parse_answer(text):
        text = str(text)
        tail = text.rsplit('####', 1)[1] if '####' in text else None
        if tail is None:
            boxed = _re.findall(r'\\boxed\{([^{}]+)\}', text)
            tail = boxed[-1] if boxed else ''
        match = _re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
        if not match:
            return None
        try:
            return str(Fraction(match.group().replace(',', '')))
        except (ValueError, ZeroDivisionError):
            return None

def _safe_numeric_expr(expr):
    """Evaluate a restricted arithmetic expression without executing arbitrary code."""
    import ast
    expr = str(expr).replace(',', '').replace('×', '*').replace('÷', '/').replace('−', '-')
    if not re.fullmatch(r'[\d\s+\-*/().]+', expr):
        return None
    try:
        node = ast.parse(expr, mode='eval')
    except (SyntaxError, ValueError):
        return None
    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = walk(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = walk(n.left), walk(n.right)
            if a is None or b is None:
                return None
            try:
                if isinstance(n.op, ast.Add): out = a + b
                elif isinstance(n.op, ast.Sub): out = a - b
                elif isinstance(n.op, ast.Mult): out = a * b
                elif isinstance(n.op, ast.Div): out = a / b
                else:
                    if abs(b) > 8 or abs(a) > 1e6: return None
                    out = a ** b
                return out if np.isfinite(out) and abs(out) < 1e12 else None
            except (ZeroDivisionError, OverflowError):
                return None
        return None
    return walk(node)

def _equation_features(text, question, answer):
    # Extract only transparent arithmetic equalities; unsupported LaTeX is ignored.
    eqs = []
    arith = re.compile(
        r'([+-]?\s*\d[\d,]*(?:\.\d+)?(?:\s*[+\-*/×÷]\s*[+-]?\s*\d[\d,]*(?:\.\d+)?)+)'
        r'\s*=\s*([+-]?\s*\d[\d,]*(?:\.\d+)?)'
    )
    frac = re.compile(
        r'\\frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*=\s*([+-]?\s*\d[\d,]*(?:\.\d+)?)'
    )
    for m in frac.finditer(text):
        a = _safe_numeric_expr(m.group(1))
        b = _safe_numeric_expr(m.group(2))
        rhs = _safe_numeric_expr(m.group(3))
        if a is not None and b not in (None, 0) and rhs is not None:
            eqs.append((a / b, rhs))
    for m in arith.finditer(text):
        lhs = _safe_numeric_expr(m.group(1))
        rhs = _safe_numeric_expr(m.group(2))
        if lhs is not None and rhs is not None:
            eqs.append((lhs, rhs))
    errors = [abs(a-b)/(abs(b)+1.0) for a,b in eqs]
    correct = [e <= 1e-3 for e in errors]
    vals = []
    answer_value = None
    if answer is not None:
        try: answer_value = float(Fraction(answer))
        except (ValueError, ZeroDivisionError, OverflowError): pass
    # Numeric tokens in the question/candidate, used only as content features.
    qnums = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', question)
    cnums = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', text)
    qset = set()
    for s in qnums:
        try: qset.add(float(Fraction(s.replace(',', ''))))
        except Exception: pass
    cvals = []
    for s in cnums:
        try: cvals.append(float(Fraction(s.replace(',', ''))))
        except Exception: pass
    overlap = sum(1 for v in cvals if any(abs(v-q) < 1e-6 for q in qset))
    answer_occurs = 0
    if answer_value is not None:
        answer_occurs = sum(abs(v-answer_value) < 1e-6 for v in cvals)
    # Candidate answer marker consistency and correction/uncertainty signals.
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    hashes = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)', text)
    marked_vals=[]
    for s in boxed + hashes:
        try: marked_vals.append(float(Fraction(s.replace(',', ''))))
        except Exception: pass
    markers = len(boxed) + len(hashes)
    unique_marked = len(set(round(v, 8) for v in marked_vals))
    lower=text.lower()
    corrections=sum(lower.count(w) for w in ('wait,','actually,','let me correct','correction:','i made a mistake','recalculate','that is incorrect','this is wrong'))
    uncertainty=sum(lower.count(w) for w in ('perhaps','maybe','not sure','could be','might be','let us check again'))
    eq_rhs_answer = 0
    last_rhs_answer = 0
    if answer_value is not None and eqs:
        eq_rhs_answer = sum(abs(rhs-answer_value) < 1e-6 for _,rhs in eqs)
        last_rhs_answer = float(abs(eqs[-1][1]-answer_value) < 1e-6)
    last_num_matches = 0
    if answer_value is not None and cvals:
        last_num_matches = float(abs(cvals[-1]-answer_value) < 1e-6)
    last200 = text[-200:]
    answer_in_tail = float(answer is not None and re.search(re.escape(answer), last200) is not None)
    return [
        np.log1p(len(eqs)), np.log1p(text.count('=')),
        len(eqs)/(text.count('=')+1e-6),
        sum(correct)/(len(correct)+1e-6),
        np.log1p(sum(not c for c in correct)),
        np.log1p(np.mean(errors) if errors else 0.0),
        np.log1p(max(errors) if errors else 0.0),
        float(bool(correct) and correct[-1]),
        float(any(not c for c in correct)),
        len(eqs)/(text.count('=')+1e-6),
        np.log1p(len(qset)), np.log1p(len(cvals)),
        overlap/(len(qnums)+1e-6), np.log1p(overlap),
        overlap/(len(cvals)+1e-6),
        float(answer_value is not None and any(abs(answer_value-q)<1e-6 for q in qset)),
        np.log1p(answer_occurs), float(last_num_matches), answer_in_tail,
        np.log1p(markers), np.log1p(len(boxed)), np.log1p(len(hashes)),
        np.log1p(unique_marked), float(unique_marked > 1),
        np.log1p(corrections), np.log1p(uncertainty),
        np.log1p(len(text.splitlines())),
        np.log1p(len(re.findall(r'[.!?](?:\s|$)', text))),
        float(bool(re.search(r'####\s*[-+]?\d', text))),
        float(bool(re.search(r'\\boxed\{', text))),
        np.log1p(eq_rhs_answer), last_rhs_answer,
        np.log1p(len(re.findall(r'\b(?:step|first|second|third|finally|therefore|thus)\b', lower))),
        np.log1p(text.count('%')), np.log1p(text.count('$')),
    ]

def handcrafted_features(bank, embeddings):
    """Return [N, 4, 46] lightweight text/numeric/within-question features."""
    n = len(bank)
    result = np.zeros((n, 4, 81), dtype=np.float32)
    for qi, row in enumerate(bank):
        candidates = row['candidates']
        answers = [parse_answer(c['text']) for c in candidates]
        counts = Counter(a for a in answers if a is not None)
        distinct = len(counts)
        norms = np.linalg.norm(embeddings[qi], axis=1) + 1e-8
        unit = embeddings[qi] / norms[:, None]
        sims = unit @ unit.T
        values = []
        for answer in answers:
            if answer is None:
                values.append(None)
            else:
                try:
                    values.append(float(Fraction(answer)))
                except (ValueError, ZeroDivisionError, OverflowError):
                    values.append(None)
        valid = [v for v in values if v is not None]

        for j, candidate in enumerate(candidates):
            text = candidate['text']
            numbers = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', text)
            words = re.findall(r"[A-Za-z]+", text)
            basic = [
                np.log1p(len(text)), np.log1p(len(words)), np.log1p(len(numbers)),
                np.log1p(sum(ch.isdigit() for ch in text)), np.log1p(text.count('=')),
                np.log1p(text.count('+')), np.log1p(text.count('-')), np.log1p(text.count('*')),
                np.log1p(text.count('/')), np.log1p(text.count('$')),
                np.log1p(text.count('####')), np.log1p(text.lower().count('answer')),
                np.log1p(text.count('\\boxed')), np.log1p(text.count('\n')),
                np.log1p(text.count('**')), np.log1p(text.count('→')),
                np.log1p(text.count('=<<')),
                float(bool(re.search(r'####\s*[-+]?\d|\\boxed\{', text))),
                float(bool(re.search(r'final answer|answer is|answer:', text, re.I))),
                float(bool(re.search(r'\btherefore\b|\bthus\b|\bso\b', text, re.I))),
            ]

            answer = answers[j]
            if answer is None:
                numeric = [0.0] * 7
            else:
                try:
                    value = float(Fraction(answer))
                    numeric = [
                        1.0, float(value < 0), math.copysign(math.log1p(abs(value)), value),
                        float(abs(value - round(value)) < 1e-9), float('/' in answer),
                        float('.' in answer), min(20, len(answer)) / 20,
                    ]
                except (ValueError, ZeroDivisionError, OverflowError):
                    numeric = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            answer_features = [
                float(answer is not None),
                counts.get(answer, 0) / 4 if answer is not None else 0.0,
                float(answer is not None and counts.get(answer, 0) == max(counts.values(), default=0)),
                distinct / 4,
                float(sum(a is not None for a in answers)) / 4,
                float(answer is not None and counts.get(answer, 0) == 1),
                float(answer is not None and any(a is not None and a != answer for a in answers)),
            ]

            other_similarities = [float(sims[j, k]) for k in range(4) if k != j]
            group_mean = embeddings[qi].mean(axis=0)
            group_mean_norm = np.linalg.norm(group_mean) + 1e-8
            relation = other_similarities + [
                float(np.mean(other_similarities)),
                float(np.max(other_similarities)),
                float(np.dot(embeddings[qi, j], group_mean) / (norms[j] * group_mean_norm)),
                float(norms[j] / (norms.mean() + 1e-8)),
                float(np.linalg.norm(embeddings[qi, j] - group_mean) / group_mean_norm),
            ]

            if answer is not None and valid:
                value = float(Fraction(answer))
                median = float(np.median(valid))
                mad = float(np.median(np.abs(np.asarray(valid) - median))) + 1e-6
                answer_relation = [
                    sum(v < value for v in valid) / len(valid),
                    float(np.sign(value - median)),
                    float(np.sign(value - median) * np.log1p(abs(value - median))),
                    float(np.log1p(abs(value - median) / mad)),
                ]
            else:
                answer_relation = [0.0] * 4

            features = basic + numeric + answer_features + relation + answer_relation + _equation_features(text, row.get('question', ''), answer)
            if len(features) != 81:
                raise RuntimeError(f'Expected 81 handcrafted features, got {len(features)}')
            result[qi, j] = np.asarray(features, dtype=np.float32)
    return result


def _resolve_checkpoint(checkpoint):
    if checkpoint is None:
        raise FileNotFoundError('This method requires a checkpoint; pass --checkpoint runs/model_bundle')
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'model.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {path}')
    return path


def predict(bank, arrays, checkpoint):
    """Return one integer candidate index (0..3) per question, in bank order."""
    checkpoint_path = _resolve_checkpoint(checkpoint)
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float32)
    if embeddings.shape != (len(bank), 4, 1024):
        raise ValueError(f'Expected embeddings shape {(len(bank), 4, 1024)}, got {embeddings.shape}')

    with np.load(checkpoint_path, allow_pickle=False) as model:
        embedding_mean = model['embedding_mean'].astype(np.float32)
        embedding_std = model['embedding_std'].astype(np.float32)
        feature_mean = model['feature_mean'].astype(np.float32)
        feature_std = model['feature_std'].astype(np.float32)
        if 'weights' in model.files:
            weights = model['weights'].astype(np.float32)
            biases = model['biases'].astype(np.float32).reshape(-1)
            scale = float(model['handcrafted_scale'].reshape(-1)[0]) if 'handcrafted_scale' in model.files else 1.5
        else:
            weights = model['weight'].astype(np.float32).reshape(1, -1)
            biases = model['bias'].astype(np.float32).reshape(-1)
            scale = float(model['handcrafted_scale'].reshape(-1)[0]) if 'handcrafted_scale' in model.files else 1.5

    embedding_features = np.clip((embeddings - embedding_mean) / embedding_std, -8, 8).astype(np.float32)
    raw_features = handcrafted_features(bank, embeddings)
    text_features = np.clip((raw_features - feature_mean) / feature_std, -8, 8).astype(np.float32)
    features = np.concatenate([embedding_features, scale * text_features], axis=-1)
    if features.shape[-1] != weights.shape[-1]:
        raise ValueError(f'Checkpoint expects {weights.shape[-1]} features, got {features.shape[-1]}')
    member_scores = np.einsum('ncf,mf->mnc', features, weights) + biases[:, None, None]
    member_scores = member_scores - member_scores.mean(axis=2, keepdims=True)
    member_scores = member_scores / (member_scores.std(axis=2, keepdims=True) + 1e-6)
    scores = member_scores.mean(axis=0)
    return np.argmax(scores, axis=1).astype(int).tolist()
