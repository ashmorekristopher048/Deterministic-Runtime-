📦 What's inside

· Tiled INT8 GEMM with DP4A hardware acceleration
· Deterministic ReLU clamp (int32 → int8)
· Simple but repeatable hash for verification
· Zero floating‑point operations – fully integer

🧠 Why this matters

· Trustless verification – same result on any GPU
· Regulatory compliance – auditable, reproducible AI
· Decentralized compute – proof of correct inference

🏗️ Architecture

· dain_int8_gemm: 32×32 tiles, shared memory, DP4A
· int_relu_clamp: clamps to [0,127] deterministically
· simple_hash: non‑cryptographic but deterministic (replace with SHA‑256 for production)

🔬 Tested on

· NVIDIA T4 (Colab)
· Should work on any CUDA‑capable GPU (SM 6.1+)

📝 License

MIT – use freely, attribution appreciated.

🙏 Author

Built with determination by Kristopher Ashmore.
GitHub Profile
# Deterministic-Runtime-
Creating a heterogenous hardware deterministic runtime
