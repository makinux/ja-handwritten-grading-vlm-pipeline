# -*- coding: utf-8 -*-
"""text-only プローブ(パネル実測パッケージ 項目 D)。

transcript_gt の**テキストのみ**から誤り有無を予測する軽量分類器(文字
bigram ナイーブベイズ、依存ライブラリなし)を訓練し、AUC を測る。

位置づけ(設計書 v2.1 §3 M6):これは**診断**であり合否閾値ではない。
テキストからの予測可能性はタスク定義上ある程度必然(誤りは式を複雑化させる
因果信号を持つ)。ただし「生成器の非現実的制約」由来のリーク(例:正解が
常に整数解)は生成器側で除去すべきであり、本プローブはそれを検出する。
最終判定は容量整合対照(同規模テキスト専用モデル vs VLM)で行う。

グループ分割: 誤りあり/なしの双子(pair_id)は必ず同じ側に入れる。
"""
import math
import random

import gen_core
from gen_core import normalize_text


def _bigrams(text):
    t = normalize_text(text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


class NaiveBayes:
    def __init__(self):
        self.counts = {0: {}, 1: {}}
        self.totals = {0: 0, 1: 0}
        self.docs = {0: 0, 1: 0}
        self.vocab = set()

    def fit(self, xs, ys):
        for x, y in zip(xs, ys):
            self.docs[y] += 1
            for g in _bigrams(x):
                self.counts[y][g] = self.counts[y].get(g, 0) + 1
                self.totals[y] += 1
                self.vocab.add(g)

    def score(self, x):
        """log P(err|x) - log P(ok|x)(定数差を除く)。"""
        v = len(self.vocab) + 1
        s = math.log(max(1, self.docs[1])) - math.log(max(1, self.docs[0]))
        for g in _bigrams(x):
            p1 = (self.counts[1].get(g, 0) + 1) / (self.totals[1] + v)
            p0 = (self.counts[0].get(g, 0) + 1) / (self.totals[0] + v)
            s += math.log(p1) - math.log(p0)
        return s

    def top_features(self, k=8):
        v = len(self.vocab) + 1
        scored = []
        for g in self.vocab:
            c1, c0 = self.counts[1].get(g, 0), self.counts[0].get(g, 0)
            if c1 + c0 < 20:
                continue
            lr = math.log((c1 + 1) / (self.totals[1] + v)) - \
                math.log((c0 + 1) / (self.totals[0] + v))
            scored.append((lr, g, c1, c0))
        scored.sort(reverse=True)
        return scored[:k]


def _auc(scores_pos, scores_neg):
    """Mann–Whitney に基づく AUC。"""
    pairs = 0.0
    wins = 0.0
    for sp in scores_pos:
        for sn in scores_neg:
            pairs += 1
            if sp > sn:
                wins += 1
            elif sp == sn:
                wins += 0.5
    return wins / pairs if pairs else 0.5


def run_probe(n_total=2000, seed=777, fraction_golds=True):
    """データ生成→pair 単位のグループ分割→NB 訓練→AUC。"""
    prev = gen_core.CONFIG["linear_fraction_golds"]
    gen_core.CONFIG["linear_fraction_golds"] = fraction_golds
    try:
        kept, _stats = gen_core.generate_batch(n_total, seed)
    finally:
        gen_core.CONFIG["linear_fraction_golds"] = prev

    rng = random.Random(seed + 1)
    data = [(r["pair_id"], r["transcript_gt"]["text"],
             0 if r["control_flag"]["error_free"] else 1)
            for r, _p in kept]
    pair_ids = sorted({d[0] for d in data})
    rng.shuffle(pair_ids)
    cut = int(len(pair_ids) * 0.7)
    train_pairs = set(pair_ids[:cut])

    tr = [(t, y) for pid, t, y in data if pid in train_pairs]
    te = [(t, y) for pid, t, y in data if pid not in train_pairs]
    nb = NaiveBayes()
    nb.fit([t for t, _ in tr], [y for _, y in tr])
    pos = [nb.score(t) for t, y in te if y == 1]
    neg = [nb.score(t) for t, y in te if y == 0]
    return {
        "fraction_golds": fraction_golds,
        "n_train": len(tr), "n_test": len(te),
        "auc": _auc(pos, neg),
        "top_error_bigrams": [(g, c1, c0)
                              for _lr, g, c1, c0 in nb.top_features(8)],
    }
