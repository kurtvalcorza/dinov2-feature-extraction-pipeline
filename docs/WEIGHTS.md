# Weight provenance and DIMER hosting

- Upstream: `timm/vit_small_patch14_dinov2.lvd142m`
- Immutable revision: `4610ca143709d58a633b6397a74412c2c3842454`
- Weight format: SafeTensors (`model.safetensors`, 88240510 bytes)
- Upstream weight license: **Apache-2.0** per the snapshot `README.md` front matter (`license: apache-2.0`) and the upstream repository. **Resolved 2026-09-12:** the original DINOv2 release (2023-04-17) was CC-BY-NC-4.0 and the upstream repository re-licensed to Apache-2.0 on 2023-08-31 (facebookresearch/dinov2, commit "Update license everywhere (#182)"; the current `LICENSE` file is the Apache License 2.0). The `cc-by-nc-4.0` string in the snapshot `config.json` `pretrained_cfg.license` is a stale field carried over from timm's April-2023 import and does not govern the weights. Both snapshot files are part of the pinned revision and both digests are in the manifest.
- Local snapshot: `weights/vit-small-dinov2/` with `dimer-base-manifest.json` (per-file bytes + SHA-256, `totalBytes` 88245132); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/dinov2_feature_extraction_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch; `from_pretrained` additionally checks the loaded model's `num_features == 384`.
- DIMER hosting: under Apache-2.0, use, modification, distribution and commercial use are permitted subject to the license and notice requirements, and DIMER may mirror the pinned checkpoint in its model store; if the `cc-by-nc-4.0` reading were the operative one, commercial use would be excluded. Resolve the discrepancy above first.
- Loader trust boundary: `timm==1.0.29` built-in `vit_small_patch14_dinov2` architecture; weights loaded from a file path via `pretrained_cfg_overlay` with `num_classes=0`; no remote code is executed. Hub download is opt-in and pinned to the revision above.
