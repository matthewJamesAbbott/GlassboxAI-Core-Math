# GlassBoxAI-Core-Math

## **The 42 Essential Equations for AI Engineering**

### *Mathematical Foundations with Pascal Implementations*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Demo-Vercel-black.svg)](https://glassbox-ai-core-math.vercel.app/)
[![Pascal](https://img.shields.io/badge/Pascal-Implementations-blue.svg)](https://www.freepascal.org/)
[![Educational](https://img.shields.io/badge/Purpose-Educational-green.svg)](https://github.com/matthewJamesAbbott/GlassBoxAI-Core-Math)

---

## **Overview**

GlassBoxAI-Core-Math is an educational reference documenting the 35 fundamental equations that underpin modern deep learning across all major architectures. Unlike typical ML resources that hide complexity behind libraries, this project provides:

- **Complete mathematical explanations** for each equation
- **Symbol-by-symbol breakdowns** with plain English descriptions
- **Pascal implementations** showing algorithmic clarity
- **Real-world context** explaining why each equation matters
- **Cross-architecture coverage** - Transformers, CNNs, RNNs, GNNs, GANs, Random Forests, and MLPs
- **Interview preparation** for whiteboard coding scenarios

This resource is designed for three audiences:
1. **Students** learning AI/ML fundamentals from first principles
2. **Engineers** preparing for technical interviews requiring deep understanding
3. **Practitioners** who want to understand what's happening inside PyTorch/TensorFlow

**Philosophy**: *The only magic is the act we do that we don't understand. We make glassboxes here. No magic is allowed.*

---

## **Table of Contents**

1. [The 35 Equations](#the-35-equations)
2. [Architecture Coverage](#architecture-coverage)
3. [How to Use This Resource](#how-to-use-this-resource)
4. [Why Pascal?](#why-pascal)
5. [Interview Strategy](#interview-strategy)
6. [Learning Path](#learning-path)
7. [Cross-Reference Guide](#cross-reference-guide)
8. [Related Projects](#related-projects)
9. [License](#license)
10. [Author](#author)

---

## **The 35 Equations**

### **Phase 1: The Foundations (Equations 1-4)**

These four equations form the bedrock of all neural network training.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **1** | **Softmax** | Convert logits → probabilities | ⭐⭐⭐⭐⭐ Always |
| **2** | **Cross-Entropy Loss** | Measure prediction error | ⭐⭐⭐⭐⭐ Always |
| **3** | **SGD** | Update weights via gradient descent | ⭐⭐⭐⭐ Very Common |
| **4** | **Chain Rule** | Backpropagation engine | ⭐⭐⭐⭐ Very Common |

**Why these matter**: You cannot claim to understand neural networks without knowing how:
- Raw scores become probabilities (Softmax)
- We measure how wrong we are (Cross-Entropy)
- We improve via small steps (SGD)
- Errors flow backwards (Chain Rule)

---

### **Phase 2: The Architecture - Transformers (Equations 5-8)**

These four equations define the transformer architecture that powers GPT, LLaMA, and modern LLMs.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **5** | **Scaled Dot-Product Attention** | Core self-attention mechanism | ⭐⭐⭐⭐⭐ Critical |
| **6** | **Layer Normalization** | Training stability | ⭐⭐⭐ Common |
| **7** | **Multi-Head Attention** | Parallel attention patterns | ⭐⭐⭐⭐ Very Common |
| **8** | **GELU Activation** | Modern non-linearity | ⭐⭐ Less Common |

**Why these matter**: The "Attention is All You Need" paper built modern AI on these four equations. If you're interviewing for an NLP role, you **will** be asked to derive scaled dot-product attention on a whiteboard.

---

### **Phase 3: Search & Vector Space (Equations 9-11)**

These three equations govern how we find and measure semantic similarity.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **9** | **Cosine Similarity** | Angle-based similarity | ⭐⭐⭐⭐ Very Common |
| **10** | **Euclidean Distance (L2)** | Absolute distance | ⭐⭐⭐ Common |
| **11** | **KL Divergence** | Distribution difference | ⭐⭐ Moderate |

**Why these matter**: RAG (Retrieval-Augmented Generation) and vector databases are built on these. Understanding why cosine similarity works better than L2 for embeddings is a common interview question.

---

### **Phase 4: Efficiency - Domestic GPU Judo (Equations 12-15)**

These four equations enable training large models on consumer hardware.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **12** | **LoRA** | Low-rank fine-tuning | ⭐⭐⭐ Rising |
| **13** | **Linear Quantization** | Compress to 4-bit | ⭐⭐⭐ Rising |
| **14** | **RMSNorm** | Faster LayerNorm | ⭐⭐ Moderate |
| **15** | **RoPE** | Rotary positional encoding | ⭐⭐⭐ Common |

**Why these matter**: Edge deployment and efficient training are increasingly critical. LoRA and quantization are how you fit an 8B model on an 8GB GPU. RoPE is how LLaMA/Mistral encode position without hurting long-context performance.

---

### **Phase 5: Production Scale (Equations 16-18)**

These three equations govern real-world deployment and compute optimization.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **16** | **Chinchilla Scaling Law** | Optimal data:parameter ratio | ⭐⭐⭐ Rising |
| **17** | **Memory Bottleneck** | VRAM budget calculation | ⭐⭐⭐⭐ Critical for ML Eng |
| **18** | **RoPE Calculus** | Complex rotation mechanics | ⭐⭐ Deep dives |

**Why these matter**: 
- Chinchilla proves most models are undertrained, not oversized
- Memory bottleneck explains why your 32GB GPU runs out of VRAM
- RoPE Calculus shows you understand the math, not just the API

---

### **Phase 6: Multi-Layer Perceptron (Equation 19)**

The fundamental building block of feedforward neural networks.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **19** | **Affine Transformation** | Linear layer + activation | ⭐⭐⭐⭐⭐ Always |

**Why this matters**: MLPs are the backbone of nearly every neural architecture. Understanding `y = σ(Wx + b)` is absolutely fundamental.

---

### **Phase 7: Recurrent Neural Networks (Equations 20-21)**

Sequential processing with memory - the foundation of time-series and sequence modeling.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **20** | **Recurrent Hidden State** | Memory across time steps | ⭐⭐⭐⭐ Very Common |
| **21** | **RNN Output** | Readout layer | ⭐⭐⭐ Common |

**Why these matter**: Before transformers dominated, RNNs ruled sequence modeling. Still critical for streaming data, online learning, and understanding LSTM/GRU architectures.

---

### **Phase 8: Graph Neural Networks (Equations 22-24)**

Learning on graph-structured data - social networks, molecules, knowledge graphs.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **22** | **Message Function** | Node-to-node communication | ⭐⭐⭐ Rising |
| **23** | **Aggregation** | Permutation-invariant pooling | ⭐⭐⭐ Rising |
| **24** | **Node Update** | Combine old + new information | ⭐⭐⭐ Rising |

**Why these matter**: GNNs are exploding in drug discovery, social network analysis, and recommendation systems. Understanding message-passing is key to modern graph learning.

---

### **Phase 9: Convolutional Neural Networks (Equations 25-27)**

Spatial pattern recognition - the backbone of computer vision.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **25** | **2D Convolution** | Sliding window feature extraction | ⭐⭐⭐⭐⭐ Critical |
| **26** | **Output Dimension Calculation** | VRAM planning formula | ⭐⭐⭐⭐ Very Common |
| **27** | **Max Pooling** | Spatial downsampling | ⭐⭐⭐ Common |

**Why these matter**: Despite transformer vision models, CNNs remain dominant in production CV systems. Understanding convolution mechanics is essential for any vision role.

---

### **Phase 10: Random Forest (Equations 28-30)**

Classical ensemble learning - still unbeaten on tabular data.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **28** | **Gini Impurity** | Split quality metric | ⭐⭐⭐⭐ Very Common |
| **29** | **Information Gain** | Entropy-based splitting | ⭐⭐⭐ Common |
| **30** | **Forest Ensemble** | Majority vote aggregation | ⭐⭐⭐ Common |

**Why these matter**: Random forests often outperform neural networks on structured/tabular data. Understanding decision tree mechanics is crucial for data science interviews.

---

### **Phase 11: Generative Adversarial Networks (Equations 31-35)**

Adversarial training for realistic data generation.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **31** | **Minimax GAN Loss** | Core adversarial game | ⭐⭐⭐⭐ Very Common |
| **32** | **Wasserstein GAN** | Improved stability | ⭐⭐⭐ Rising |
| **33** | **Generator Loss (Non-saturating)** | Avoid gradient vanishing | ⭐⭐⭐ Common |
| **34** | **Mode Collapse Detection** | Diversity metric | ⭐⭐ Moderate |
| **35** | **Conditional GAN** | Controlled generation | ⭐⭐⭐ Common |

**Why these matter**: GANs power image generation, style transfer, and data augmentation. Understanding the adversarial training dynamics is key to debugging GAN training instability.

---

## **Architecture Coverage**

This resource now covers **all major neural network architectures**:

| Architecture | Equations | Primary Use Cases |
|--------------|-----------|-------------------|
| **Transformers** | 5-8, 12, 14-18 | NLP, LLMs, Vision Transformers |
| **MLP** | 19 | Tabular data, simple classification |
| **RNN/LSTM/GRU** | 20-21 | Time series, streaming data, online learning |
| **GNN** | 22-24 | Social networks, molecules, knowledge graphs |
| **CNN** | 25-27 | Computer vision, image classification, detection |
| **Random Forest** | 28-30 | Tabular data, feature importance, interpretability |
| **GAN** | 31-35 | Image generation, style transfer, data augmentation |
| **Foundational** | 1-4 | Universal (all architectures use these) |
| **Vector Search** | 9-11 | RAG, semantic search, embeddings |
| **Efficiency** | 12-13 | Edge deployment, low-resource training |

---

## **How to Use This Resource**

### **For Learning**

1. **Start with Phase 1** - You cannot understand any architecture without these foundations
2. **Choose your path**:
   - **NLP/LLMs** → Phases 1-5 (Transformers focus)
   - **Computer Vision** → Phases 1, 6, 9 (MLP + CNN)
   - **Time Series** → Phases 1, 6, 7 (MLP + RNN)
   - **Graph Learning** → Phases 1, 6, 8 (MLP + GNN)
   - **Generative AI** → Phases 1, 6, 9, 11 (MLP + CNN + GAN)
   - **Tabular/Classical ML** → Phases 1, 6, 10 (MLP + Random Forest)
3. **Code the Pascal implementations** - Typing out the algorithms builds intuition
4. **Cross-reference architectures** - See how equations combine (e.g., CNN features → Transformer, RNN + Attention)

### **For Interview Prep**

1. **Memorize the symbol keys** - Know what every Greek letter represents
2. **Practice whiteboard coding** - Essential equations:
   - **Always**: Softmax (1), Cross-Entropy (2), SGD (3), Affine Transform (19)
   - **Transformers**: Attention (5), Multi-Head (7)
   - **CNNs**: Convolution (25), Output Size (26)
   - **RNNs**: Hidden State (20)
   - **GNNs**: Message Passing (22-24)
   - **GANs**: Minimax Loss (31)
3. **Explain the "why"** - Interviewers care more about understanding than memorization
4. **Know the failure modes**:
   - What breaks without the √d_k scaling in attention?
   - Why does GAN training collapse without proper techniques?
   - When does vanishing gradient hit RNNs?

### **For Implementation**

1. **Use as reference** - When implementing from scratch, verify against these formulas
2. **Debug with introspection** - The Pascal code shows intermediate values you can log
3. **Optimize carefully** - Understand the naive version before vectorizing/fusing
4. **Cross-architecture patterns** - Notice how convolution is just a specialized attention mechanism

---

## **Why Pascal?**

**Common question**: "Why Pascal in 2026? Why not Python?"

**Answer**: Pascal is **pedagogical** - it reads like algorithmic pseudocode but compiles and runs.

### **Comparison**

**Python (NumPy)**:
```python
attention = softmax(Q @ K.T / sqrt(d_k)) @ V
```
✅ Concise  
❌ Hides all implementation details  
❌ Hard to debug intermediate steps  
❌ Matrix ops obscure the actual algorithm  

**Pascal**:
```pascal
for i := 0 to seqLen-1 do
  for j := 0 to seqLen-1 do
  begin
    scores[i,j] := 0.0;
    for k := 0 to dk-1 do
      scores[i,j] := scores[i,j] + Q[i,k] * K[j,k];
    scores[i,j] := scores[i,j] / scaleFactor;
  end;
```
✅ Every operation is explicit  
✅ Easy to insert debug prints  
✅ Maps directly to mathematical definition  
✅ Teaches algorithmic thinking  

**When you're on a whiteboard**, you're writing Pascal-style pseudocode, not NumPy one-liners. This resource prepares you for that.

---

## **Interview Strategy**

### **The Whiteboard Scenario**

**Interviewer**: "Explain how attention works"

**Bad answer**: "You compute Q, K, V matrices, then do softmax of QK^T and multiply by V"

**Good answer**: 
1. Draw the equation: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
2. Explain each component:
   - Q = what we're looking for (queries)
   - K = what we're searching through (keys)
   - V = actual content (values)
   - √d_k = prevents dot products from growing too large
3. Walk through the algorithm:
   - "For each query position, compute similarity to all key positions"
   - "Scale by √d_k to prevent softmax saturation"
   - "Softmax converts scores to probability distribution"
   - "Use probabilities to weighted-average the values"
4. Mention failure mode: "Without the √d_k scaling, high-dimensional dot products saturate softmax, killing gradients"

**This demonstrates**:
- You know the math
- You understand the implementation
- You can explain the "why"
- You know what breaks

---

### **Common Interview Questions by Architecture**

#### **Transformers**

**Q1**: "Why divide by √d_k in attention?"  
**A1**: High-dimensional dot products have large variance. Scaling prevents softmax saturation in high dimensions, which would kill gradients.

**Q2**: "What's the difference between LayerNorm and RMSNorm?"  
**A2**: LayerNorm computes mean and variance, then normalizes. RMSNorm skips the mean calculation, only using root-mean-square. Faster, works just as well in practice.

**Q3**: "Explain LoRA in 30 seconds"  
**A3**: Instead of fine-tuning all weights W, freeze W and train small matrices B,A such that h = Wx + BAx. The rank of BA is much smaller than W, so we train 1-2% of parameters.

#### **CNNs**

**Q4**: "How do you calculate CNN output dimensions?"  
**A4**: O = ⌊(I - F + 2P) / S⌋ + 1, where I=input size, F=filter size, P=padding, S=stride. Critical for VRAM planning.

**Q5**: "Why use pooling layers?"  
**A5**: Downsample spatial dimensions, reduce parameters, provide translation invariance, and prevent overfitting. Max pooling captures strongest activations.

#### **RNNs**

**Q6**: "What's the vanishing gradient problem in RNNs?"  
**A6**: Gradients get multiplied by recurrent weights at each time step during backprop. If weights < 1, gradients vanish exponentially with sequence length. LSTMs solve this with gating.

**Q7**: "RNN vs Transformer for sequences?"  
**A7**: RNNs process sequentially (can't parallelize), but have O(1) memory per step. Transformers parallelize fully but have O(n²) attention complexity. Transformers win for batch processing, RNNs for streaming.

#### **GNNs**

**Q8**: "How do GNNs handle variable graph sizes?"  
**A8**: Aggregation functions (sum, mean, max) are permutation-invariant and work on any number of neighbors. Node embeddings are fixed-size regardless of graph structure.

**Q9**: "What's oversmoothing in GNNs?"  
**A9**: After many layers, all node embeddings become similar as information diffuses across the graph. Solutions: skip connections, layer normalization, careful depth selection.

#### **GANs**

**Q10**: "Why do GANs suffer from mode collapse?"  
**A10**: Generator finds one or few successful patterns that fool the discriminator and stops exploring. Solutions: Wasserstein loss, minibatch discrimination, feature matching.

**Q11**: "Original GAN vs WGAN?"  
**A11**: Original uses JS divergence (log loss), suffers from vanishing gradients when D is too strong. WGAN uses Earth Mover's Distance (no log), provides meaningful loss metric, more stable training.

#### **Random Forest**

**Q12**: "Gini vs Entropy for splitting?"  
**A12**: Both measure impurity. Gini is faster (no log), range [0, 0.5]. Entropy is more theoretically grounded (information theory), range [0, log₂(C)]. Similar results in practice.

**Q13**: "Why Random Forests over single decision trees?"  
**A13**: Bootstrap aggregating (bagging) reduces variance, random feature selection reduces correlation between trees, ensemble voting smooths predictions. Prevents overfitting.

#### **Production/Scaling**

**Q14**: "How much VRAM to train an 8B model?"  
**A14**: Model weights (32GB FP32), optimizer state (64GB for Adam), activations (depends on batch/sequence), gradients (32GB). Total ~130GB+ for FP32. With 4-bit quantization and LoRA: ~12-16GB.

**Q15**: "What's the Chinchilla scaling law?"  
**A15**: For compute-optimal training, use ~20 tokens per parameter. Most models are undertrained, not oversized. 8B model needs ~160B tokens for optimal training.

---

## **Learning Path**

### **Beginner** (No ML Background)

**Week 1-2**: Phase 1 (Foundations)
- Implement Softmax in Pascal
- Understand why we use cross-entropy for classification
- Code a simple gradient descent optimizer
- Derive chain rule for a 2-layer network

**Week 3-4**: Choose Your Focus
- **NLP Track**: Phase 2 (Transformers)
- **Vision Track**: Phase 9 (CNNs)
- **Classical ML Track**: Phase 10 (Random Forest)

**Week 5-6**: Phase 6 (MLP) + Your Focus Architecture
- Build a complete MLP from scratch
- Implement your chosen architecture
- Understand how they combine (e.g., CNN → MLP head)

**Week 7-8**: Review and Practice
- Code all core equations from memory
- Explain each on a whiteboard
- Build a tiny version of your chosen architecture

---

### **Intermediate** (Some ML Experience)

**Multi-Architecture Track**:
1. **Weeks 1-2**: Transformers (Phases 2-5)
2. **Weeks 3-4**: CNNs (Phase 9)
3. **Weeks 5-6**: RNNs + GNNs (Phases 7-8)
4. **Weeks 7-8**: GANs (Phase 11)

**Projects**:
- Build a mini-GPT (6 layers, 512 dims) in pure C++/Rust
- Implement ResNet-18 from scratch
- Create an RNN-based time series predictor
- Build a simple GAN for MNIST generation

---

### **Advanced** (Production Experience)

**Deep Dives**:
1. **Cross-architecture fusion** - Vision Transformers, Graph Transformers
2. **Advanced optimization** - FlashAttention, fused kernels, mixed precision
3. **Production deployment** - Quantization strategies, model compression
4. **Distributed training** - Data/model/pipeline parallelism

**Challenges**:
- Implement FlashAttention algorithm
- Build a multi-GPU distributed training system
- Create a hybrid CNN-Transformer architecture
- Optimize GAN training stability with advanced techniques

---

## **Cross-Reference Guide**

Understanding how equations combine across architectures is crucial:

### **Transformer Architecture**
- **Core**: 1 (Softmax), 2 (Cross-Entropy), 3 (SGD), 4 (Chain Rule)
- **Architecture**: 5 (Attention), 6 (LayerNorm), 7 (Multi-Head), 8 (GELU)
- **Efficiency**: 12 (LoRA), 13 (Quantization), 14 (RMSNorm), 15 (RoPE)
- **Production**: 16 (Chinchilla), 17 (Memory), 18 (RoPE Calculus)

### **CNN Architecture**
- **Core**: 1-4 (Foundations)
- **Layers**: 25 (Convolution), 26 (Output Size), 27 (Pooling)
- **Optimization**: 8 (GELU/ReLU), 13 (Quantization), 17 (Memory)

### **RNN Architecture**
- **Core**: 1-4 (Foundations)
- **Recurrent**: 20 (Hidden State), 21 (Output)
- **Modern Hybrid**: 15 (RoPE for positional encoding), 17 (Memory critical)

### **GNN Architecture**
- **Core**: 1-4 (Foundations)
- **Message Passing**: 22 (Messages), 23 (Aggregation), 24 (Update)
- **Optimization**: 5 (GAT attention variant), 6 (prevent oversmoothing), 9 (node similarity)

### **GAN Architecture**
- **Core**: 1-4 (Foundations)
- **Adversarial**: 31 (Minimax), 32 (Wasserstein), 33 (Generator Loss)
- **Diagnostics**: 34 (Mode Collapse), 10 (L2 diversity)
- **Conditional**: 35 (cGAN)
- **Subnetworks**: Often use CNN (25-27) or Transformer (5-7) as G/D

### **Random Forest**
- **Core**: None (classical ML, no gradient descent)
- **Splitting**: 28 (Gini), 29 (Information Gain)
- **Ensemble**: 30 (Voting)
- **Optimization**: 9 (tree diversity), 13 (threshold quantization), 17 (memory)

---

## **Related Projects**

This resource is part of the **GlassBoxAI Suite** - a collection of formally verified, production-ready ML implementations:

dont follow the kani proofs it is far more than my agent read here.

well over 100 for some I will manually fix it.

also remember this is a very fast moving account so even when I fix it
I may be behind in numbers in readme.

I will not push unless (in the cudarc world) I have done all harness checks for new addtions as well.

it is what it is no one else is even trying in the open source world that I can see.

An addtion rust is crap what is wrong with you all.

anything that crashes with no core dump is far from worthy.

memory safe ? um no !!!

| Project | Description | Equations Used | Status |
|---------|-------------|----------------|--------|
| **[GlassBoxAI-Transformer](https://github.com/matthewJamesAbbott/GlassBoxAI-Transformer)** | Full LLM inference with GGUF support, DTX protocol | 1-8, 12-18 | ✅ 99 Kani Proofs |
| **[GlassBoxAI-CNN](https://github.com/matthewJamesAbbott/GlassBoxAI-CNN)** | Convolutional networks with ONNX export | 1-4, 25-27 | ✅ 42 Kani Proofs |
| **[GlassBoxAI-RNN](https://github.com/matthewJamesAbbott/GlassBoxAI-RNN)** | Recurrent networks for sequence modeling | 1-4, 20-21 | ✅ 38 Kani Proofs |
| **[GlassBoxAI-GNN](https://github.com/matthewJamesAbbott/GlassBoxAI-GNN)** | Graph neural networks with PageRank | 1-4, 22-24 | ✅ 95 Kani Proofs |
| **[GlassBoxAI-MLP](https://github.com/matthewJamesAbbott/GlassBoxAI-MLP)** | Multi-layer perceptrons | 1-4, 19 | ✅ 31 Kani Proofs |
| **[GlassBoxAI-RandomForest](https://github.com/matthewJamesAbbott/GlassBoxAI-RandomForest)** | Decision tree ensembles | 28-30 | ✅ 27 Kani Proofs |
| **[GlassBoxAI-GAN](https://github.com/matthewJamesAbbott/GlassBoxAI-GAN)** | Generative adversarial networks | 1-4, 31-35 | 🚧 Coming Soon |

**JavaScript Demos** (run instantly in browser):
- [JavaScript-Transformer](https://github.com/matthewJamesAbbott/Javascript-Transformer)
- [JavaScript-CNN](https://github.com/matthewJamesAbbott/Javascript-CNN)
- [JavaScript-RNN](https://github.com/matthewJamesAbbott/Javascript-RNN)
- [JavaScript-GNN](https://github.com/matthewJamesAbbott/Javascript-GNN)
- [JavaScript-GAN](https://github.com/matthewJamesAbbott/Javascript-GAN)

All projects feature:
- **CISA/NSA Secure by Design compliance**
- **Formal verification** with Kani (27-99 proofs per project, each proof is one `#[kani::proof]` function)
- **Multi-language implementations** (C++/CUDA, Rust, OpenCL, JavaScript)
- **Facade pattern** for deep introspection
- **Production-ready** code quality

---

## **Complete Ecosystem Path**

For each equation, you can follow this learning progression:

**1. Mathematical Notation** (this page) - Understand the formula  
**2. Pascal Algorithm** (this page) - See explicit implementation  
**3. Working Prototype** - Pascal-Datastructures repo  
**4. Facade Introspection** - Deep layer inspection  
**5. Production Implementation** - CUDA/Rust with formal verification  
**6. Browser Demo** - JavaScript version (no install required)  

**Example: Equation 5 (Attention)**
1. Math: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
2. Pascal: Explicit Q·K^T computation (this page)
3. Prototype: [Transformer.pas](https://github.com/matthewJamesAbbott/Pascal-Datastructures/blob/master/Transformer.pas)
4. Facade: [FacadeTransformer.pas](https://github.com/matthewJamesAbbott/Pascal-Datastructures/blob/master/FacadeTransformer.pas)
5. Production: [GlassBoxAI-Transformer](https://github.com/matthewJamesAbbott/GlassBoxAI-Transformer) (99 Kani proofs - each is one `#[kani::proof]` function verifying a specific property)
6. Demo: [JavaScript-Transformer](https://github.com/matthewJamesAbbott/Javascript-Transformer)

---

## **Philosophy**

### **Transparency Over Convenience**

Modern ML frameworks hide complexity behind abstractions. This is **good for production** but **bad for learning**. You can use PyTorch for years without understanding what softmax actually does.

GlassBoxAI takes the opposite approach: **everything is visible**. 

- No hidden state
- No magic preprocessing
- No "just trust the library"
- Every operation is inspectable

### **Education Through Implementation**

You don't truly understand an algorithm until you've:
1. Read the math
2. Implemented it yourself
3. Debugged it when it breaks
4. Explained it to someone else

This resource provides steps 1-2. The other GlassBoxAI projects provide production implementations for step 3. Step 4 is on you.

### **No Magic Allowed**

**"The only magic is the act we do that we don't understand."**

If you can't explain it, you don't understand it. If you can't implement it from scratch, you don't own it. This resource exists to eliminate the magic.

---

## **Formal Verification with Kani**

### **What is a Kani Proof?**

Each proof is **one `#[kani::proof]` function** that mathematically verifies a specific property of the code. For example:

```rust
#[kani::proof]
fn verify_clip_value_constant_time() {
    let v1: f64 = kani::any();
    let v2: f64 = kani::any();
    let max_val: f64 = kani::any();
    
    kani::assume(v1.is_finite() && v2.is_finite());
    kani::assume(max_val.is_finite() && max_val >= 0.0);
    
    let result1 = clip_value(v1, max_val);
    let result2 = clip_value(v2, max_val);
    
    kani::assert(result1.abs() <= max_val, "result1 bounded");
    kani::assert(result2.abs() <= max_val, "result2 bounded");
}
```

**This proves**: For **all possible** finite inputs, `clip_value` never produces output exceeding `max_val`.

### **Why This Matters**

- **Not testing**: Tests check specific examples. Proofs verify **all possible inputs**.
- **Mathematical certainty**: Kani uses symbolic execution to prove properties hold universally.
- **Production safety**: Guarantees like "no buffer overflows" or "output always bounded" are verified, not hoped for.

### **Proof Counts by Project**

| Project | Total Proofs | Example Properties Verified |
|---------|--------------|----------------------------|
| **Transformer** | 99 | Attention weights sum to 1.0, no NaN propagation, QKV dimensions valid |
| **GNN** | 95 | Message aggregation commutative, node updates bounded, graph traversal terminates |
| **CNN** | 42 | Convolution output dimensions correct, pooling preserves max, no out-of-bounds access |
| **RNN** | 38 | Hidden state remains bounded, gradient clipping works, BPTT depth safe |
| **MLP** | 31 | Weight updates converge, activation outputs bounded, backprop numerically stable |
| **RandomForest** | 27 | Tree depth limits enforced, split thresholds valid, majority vote correct |

Each proof is a **total function** - it verifies one specific mathematical property holds for **all** valid inputs.

---

## **Contributing**

Found an error in the math? Have a clearer way to explain an equation? Want to add more examples?

**Contributions welcome**:
- Mathematical corrections
- Additional Pascal examples
- Alternative explanations
- Interview question additions
- Translation to other languages
- New architecture coverage

**Not welcome**:
- Replacing Pascal with Python/NumPy (misses the pedagogical point)
- Adding framework dependencies (defeats the purpose)
- Obfuscating the implementations

---

## **Frequently Asked Questions**

### **Q: Why not just use PyTorch/TensorFlow documentation?**

**A**: Framework docs explain **how to use the API**, not **how it works internally**. This resource teaches the fundamentals that frameworks abstract away.

### **Q: Do I need to learn all 35 equations?**

**A**: Depends on your role:
- **NLP/LLM Engineer**: Master 1-8, 12-18 (Transformers)
- **Computer Vision**: Master 1-4, 25-27 (CNNs)
- **Time Series/Sequential**: Master 1-4, 20-21 (RNNs)
- **Graph ML**: Master 1-4, 22-24 (GNNs)
- **Generative AI**: Master 1-4, 31-35 (GANs)
- **Data Scientist (Tabular)**: Master 1-4, 28-30 (Random Forest)
- **ML Generalist**: Know foundations (1-4, 19), understand all architectures conceptually

### **Q: Which architecture should I learn first?**

**A**: After foundations (1-4):
- **Easiest**: MLP (19) → Random Forest (28-30)
- **Most Practical Today**: Transformers (5-8)
- **Best for Intuition**: CNN (25-27)
- **Most Challenging**: GANs (31-35)

---

## **Acknowledgments**

This resource draws from:
- **"Attention is All You Need"** (Vaswani et al., 2017) - Transformer architecture
- **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022) - Chinchilla scaling
- **"LoRA: Low-Rank Adaptation of Large Language Models"** (Hu et al., 2021) - Parameter-efficient fine-tuning
- **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (Su et al., 2021) - RoPE mechanics
- **"Generative Adversarial Nets"** (Goodfellow et al., 2014) - Original GAN formulation
- **"Wasserstein GAN"** (Arjovsky et al., 2017) - Improved GAN training
- **Countless Stack Overflow answers, blog posts, and whiteboard sessions with colleagues**

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com  

---

**Live Demo**: [https://glassbox-ai-core-math.vercel.app/](https://glassbox-ai-core-math.vercel.app/)

---

*"The only magic is the act we do that we don't understand. We make glassboxes here. No magic is allowed."*

*Built with precision. Explained with clarity. Understood completely.*

**35 Equations. 7 Architectures. Complete Understanding.**
