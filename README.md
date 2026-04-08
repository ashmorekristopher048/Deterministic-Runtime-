 


· Tiled INT8 GEMM with DP4A hardware acceleration
· Deterministic ReLU clamp (int32 → int8)
· Simple but repeatable hash for verification
· Zero floating‑point operations – fully integer


· Trustless verification – same result on any GPU
· Regulatory compliance – auditable, reproducible AI
· Decentralized compute – proof of correct inferenceArchitecture

· dain_int8_gemm: 32×32 tiles, shared memory, DP4A
· int_relu_clamp: clamps to [0,127] deterministically
· proven repeatable merkle tree root on latest version

Tested on

· NVIDIA T4 (Colab)
· Should work on any CUDA‑capable GPU (SM 6.1+)
License

MIT – use freely, attribution appreciated.
 Author

Built with determination by Kristopher Ashmore.
GitHub Profile
# Deterministic-Runtime-
Creating a heterogenous hardware deterministic runtime
