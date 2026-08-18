実装とテスト更新までは完了しました。

- problem 行を常にフォント描画へ変更
- problem 行を fallback 率集計から除外
- ETL9G を p95 stretch＋γ=0.75で正規化
- 階調保持・problem 行固定の回帰テストを追加
- テスト結果: **10/10 PASS、SKIP 0**
- コミットなし

ただし直後に Windows の sandbox helper が `0xc0000142 (DLL initialization failed)` で起動不能になり、以下は未完了です。

- `out/etl_demo_page.png` の再生成
- `out/etl_glyph_contrast_check.png` の生成・目視確認
- 最終 `git diff --check`

ワークスペース実行環境の再起動後に続行が必要です。