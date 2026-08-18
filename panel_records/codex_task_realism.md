# 実装タスク: 手続き的リアリズム層(設計書 §3 末尾・実行順序表 優先9)

目的: ETL グリフバンク描画(glyph_source="etl")の「字間・ベースラインの不連続感」を、diffusion なしの決定論的変形で解消する。パネル確定条件: (a) 各変形後に mask 基準で bbox を再計算する、(b) 変形強度に仕様化された上限を置く(強変形は字形ラベル自体を壊す)、(c) 双子でパラメータを完全共有する(疑似筆者ID・文字IDベース)。

## 実装(pipeline/realism.py 新規 + m3_render 統合)

`render_record(..., realism=None|dict)` を追加(既定 None=挙動不変)。realism 有効時、以下を適用:

1. **ベースラインスプライン**: 行ごとに疑似筆者IDから導出した低周波スプライン(振幅上限 ±6px、波長 300px 以上)で文字の y をシフト。行内で連続。
2. **サイズ・傾きドリフト**: 文字サイズを行内で緩やかに変動(±8%上限)、グリフを微回転(±4°上限)。seed は (pseudo_writer_id, key, stable_char_id)。
3. **弾性変形(グリフ単位)**: グリフの L mask に低解像度変位場(格子 4x4、変位上限=グリフ幅の 4%)によるワープ。上限は定数 REALISM_LIMITS に集約し docstring で「字形ラベル保護のための上限」と明記。
4. **インク滲み・かすれ**: グリフ mask に対し、(i) 軽い膨張+ガウスで滲み、(ii) 乱数ノイズ mask との乗算でかすれ、を筆者ごとの強度(上限固定)で。階調保持。
5. **字間の自然化**: grid セル内で文字のx位置を ±pitch*0.06 まで筆者依存にオフセット(セル自体は動かさない=アンカー不変)。

制約:
- すべての乱数は _seed_from_parts 流儀で (pseudo_writer_id または pair_id, key, stable_char_id, 変形名) から導出。逐次消費禁止。
- bbox は変形後の最終 mask から再計算(既存の ink-mask 方式・マージン2px)。
- 出力メタデータに realism パラメータの要約(適用有無・強度プロファイルID)を追加。
- font モードでも動くこと(グリフ mask 経由の共通経路に置く)。

## テスト(test_m3_render_fix.py に追加)

(a) realism 有効の双子で変異スパン外の bbox 完全一致、(b) 同一レコード2回で PNG SHA-256 一致、(c) 変形後 bbox のインク質量包含(既存基準と同じ 0.1% 未満が bbox 外)、(d) REALISM_LIMITS を超える設定を渡すと ValueError。既存10テストも PASS のまま。

## デモ

out/dataset.jsonl の最初の error_free=false レコードで、(realism なし / あり) の2枚を out/realism_demo_off.png / out/realism_demo_on.png に生成(glyph_source="etl")。

完了条件: 全テスト PASS、デモ2枚生成、変更ファイルと要件対応の報告。コミットはしない。
