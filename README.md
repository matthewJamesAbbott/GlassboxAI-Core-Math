# GlassBoxAI-Core-Math

## **The 18 Essential Equations for AI Engineering**

### *Mathematical Foundations with Pascal Implementations*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Demo-Vercel-black.svg)](https://glassbox-ai-core-math.vercel.app/)
[![Pascal](https://img.shields.io/badge/Pascal-Implementations-blue.svg)](https://www.freepascal.org/)
[![Educational](https://img.shields.io/badge/Purpose-Educational-green.svg)](https://github.com/matthewJamesAbbott/GlassBoxAI-Core-Math)

---

## **Overview**

GlassBoxAI-Core-Math is an educational reference documenting the 18 fundamental equations that underpin modern deep learning and transformer architectures. Unlike typical ML resources that hide complexity behind libraries, this project provides:

- **Complete mathematical explanations** for each equation
- **Symbol-by-symbol breakdowns** with plain English descriptions
- **Pascal implementations** showing algorithmic clarity
- **Real-world context** explaining why each equation matters
- **Interview preparation** for whiteboard coding scenarios

This resource is designed for three audiences:
1. **Students** learning AI/ML fundamentals from first principles
2. **Engineers** preparing for technical interviews requiring deep understanding
3. **Practitioners** who want to understand what's happening inside PyTorch/TensorFlow

**Philosophy**: *The only magic is the act we do that we don't understand. We make glassboxes here. No magic is allowed.*

---

## **Table of Contents**

1. [The 18 Equations](#the-18-equations)
2. [How to Use This Resource](#how-to-use-this-resource)
3. [Why Pascal?](#why-pascal)
4. [Interview Strategy](#interview-strategy)
5. [Learning Path](#learning-path)
6. [Technical Background](#technical-background)
7. [Related Projects](#related-projects)
8. [License](#license)
9. [Author](#author)

---

## **The 18 Equations**

### **Phase 1: The Foundations**

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

### **Phase 2: The Architecture**

These four equations define the transformer architecture that powers GPT, LLaMA, and modern LLMs.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **5** | **Scaled Dot-Product Attention** | Core self-attention mechanism | ⭐⭐⭐⭐⭐ Critical |
| **6** | **Layer Normalization** | Training stability | ⭐⭐⭐ Common |
| **7** | **Multi-Head Attention** | Parallel attention patterns | ⭐⭐⭐⭐ Very Common |
| **8** | **GELU Activation** | Modern non-linearity | ⭐⭐ Less Common |

**Why these matter**: The "Attention is All You Need" paper built modern AI on these four equations. If you're interviewing for an NLP role, you **will** be asked to derive scaled dot-product attention on a whiteboard.

---

### **Phase 3: Search & Vector Space**

These three equations govern how we find and measure semantic similarity.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **9** | **Cosine Similarity** | Angle-based similarity | ⭐⭐⭐⭐ Very Common |
| **10** | **Euclidean Distance (L2)** | Absolute distance | ⭐⭐⭐ Common |
| **11** | **KL Divergence** | Distribution difference | ⭐⭐ Moderate |

**Why these matter**: RAG (Retrieval-Augmented Generation) and vector databases are built on these. Understanding why cosine similarity works better than L2 for embeddings is a common interview question.

---

### **Phase 4: Efficiency (Domestic GPU Judo)**

These four equations enable training large models on consumer hardware.

| # | Equation | Purpose | Interview Frequency |
|---|----------|---------|-------------------|
| **12** | **LoRA** | Low-rank fine-tuning | ⭐⭐⭐ Rising |
| **13** | **Linear Quantization** | Compress to 4-bit | ⭐⭐⭐ Rising |
| **14** | **RMSNorm** | Faster LayerNorm | ⭐⭐ Moderate |
| **15** | **RoPE** | Rotary positional encoding | ⭐⭐⭐ Common |

**Why these matter**: Edge deployment and efficient training are increasingly critical. LoRA and quantization are how you fit an 8B model on an 8GB GPU. RoPE is how LLaMA/Mistral encode position without hurting long-context performance.

---

### **Phase 5: Production Scale (God's Work)**

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

## **How to Use This Resource**

### **For Learning**

1. **Start with Phase 1** - You cannot understand transformers without these foundations
2. **Code the Pascal implementations** - Typing out the algorithms builds intuition
3. **Modify the examples** - Change dimensions, add debug prints, break things intentionally
4. **Work backwards from equations** - Given an equation, derive the Pascal code yourself

### **For Interview Prep**

1. **Memorize the symbol keys** - Know what every Greek letter represents
2. **Practice whiteboard coding** - Write Softmax, Attention, and SGD from memory
3. **Explain the "why"** - Interviewers care more about understanding than memorization
4. **Know the failure modes** - What breaks without the √d_k scaling? Why epsilon in LayerNorm?

### **For Implementation**

1. **Use as reference** - When implementing from scratch, verify against these formulas
2. **Debug with introspection** - The Pascal code shows intermediate values you can log
3. **Optimize carefully** - Understand the naive version before vectorizing/fusing

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

### **Common Interview Questions**

**Q1**: "Why divide by √d_k in attention?"  
**A1**: High-dimensional dot products have large variance. Scaling prevents softmax saturation in high dimensions, which would kill gradients.

**Q2**: "What's the difference between LayerNorm and RMSNorm?"  
**A2**: LayerNorm computes mean and variance, then normalizes. RMSNorm skips the mean calculation, only using root-mean-square. Faster, works just as well in practice.

**Q3**: "Explain LoRA in 30 seconds"  
**A3**: Instead of fine-tuning all weights W, freeze W and train small matrices B,A such that h = Wx + BAx. The rank of BA is much smaller than W, so we train 1-2% of parameters.

**Q4**: "How much VRAM to train an 8B model?"  
**A4**: Model weights (32GB FP32), optimizer state (64GB for Adam), activations (depends on batch/sequence), gradients (32GB). Total ~130GB+ for FP32. With 4-bit quantization and LoRA: ~12-16GB.

---

## **Learning Path**

### **Beginner** (No ML Background)

**Week 1-2**: Phase 1 (Foundations)
- Implement Softmax in Pascal
- Understand why we use cross-entropy for classification
- Code a simple gradient descent optimizer
- Derive chain rule for a 2-layer network

**Week 3-4**: Phase 2 (Architecture)
- Implement scaled dot-product attention
- Build a single attention head
- Extend to multi-head attention
- Compare ReLU vs GELU

**Week 5-6**: Phase 3 (Vector Space)
- Implement cosine similarity for text embeddings
- Build a simple vector search
- Understand when L2 vs cosine matters

**Week 7-8**: Review and Practice
- Code all 15 core equations from memory
- Explain each on a whiteboard
- Implement a tiny transformer (2 layers, 64 dims)

---

### **Intermediate** (Some ML Experience)

**Focus Areas**:
1. **Attention mechanisms** - Implement from scratch, no libraries
2. **Quantization** - Understand 4-bit, 8-bit quantization schemes
3. **Memory optimization** - Calculate VRAM for different model sizes
4. **LoRA mathematics** - Derive why low-rank works

**Projects**:
- Build a mini-GPT (6 layers, 512 dims) in pure C++/Rust
- Implement 4-bit quantization with proper scale/zero-point
- Create a LoRA fine-tuning pipeline

---

### **Advanced** (Production Experience)

**Deep Dives**:
1. **RoPE mathematics** - Complex rotation theory
2. **Chinchilla optimality** - Compute-optimal training
3. **KV-cache optimization** - How to speed up inference
4. **FlashAttention** - Fused attention kernels

**Challenges**:
- Implement FlashAttention algorithm
- Derive optimal model size for compute budget
- Build a distributed training pipeline

---

## **Technical Background**

### **What This Covers**

✅ **Transformer architecture** (equations 5-8, 15, 18)  
✅ **Training fundamentals** (equations 1-4)  
✅ **Vector search/RAG** (equations 9-11)  
✅ **Efficiency techniques** (equations 12-14)  
✅ **Production scaling** (equations 16-17)  

### **What This Doesn't Cover**

❌ Convolutional networks (see GlassBoxAI-CNN)  
❌ Recurrent networks (see GlassBoxAI-RNN)  
❌ Graph neural networks (see GlassBoxAI-GNN)  
❌ Reinforcement learning  
❌ Mixture of Experts (coming soon)  

---

## **Related Projects**

This resource is part of the **GlassBoxAI Suite** - a collection of formally verified, production-ready ML implementations:

| Project | Description | Status |
|---------|-------------|--------|
| **[GlassBoxAI-Transformer](https://github.com/matthewJamesAbbott/GlassBoxAI-Transformer)** | Full LLM inference with GGUF support, DTX protocol | ✅ Stable |
| **[GlassBoxAI-CNN](https://github.com/matthewJamesAbbott/GlassBoxAI-CNN)** | Convolutional networks with ONNX export | ✅ Stable |
| **[GlassBoxAI-RNN](https://github.com/matthewJamesAbbott/GlassBoxAI-RNN)** | Recurrent networks for sequence modeling | ✅ Stable |
| **[GlassBoxAI-GNN](https://github.com/matthewJamesAbbott/GlassBoxAI-GNN)** | Graph neural networks with PageRank | ✅ Stable |
| **[GlassBoxAI-MLP](https://github.com/matthewJamesAbbott/GlassBoxAI-MLP)** | Multi-layer perceptrons | ✅ Stable |
| **[GlassBoxAI-RandomForest](https://github.com/matthewJamesAbbott/GlassBoxAI-RandomForest)** | Decision tree ensembles | ✅ Stable |

All projects feature:
- **CISA/NSA Secure by Design compliance**
- **Formal verification** with Kani (40-99 proofs per project)
- **Multi-language implementations** (C++/CUDA, Rust, OpenCL, JavaScript)
- **Facade pattern** for deep introspection
- **Production-ready** code quality

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

## **Contributing**

Found an error in the math? Have a clearer way to explain an equation? Want to add more examples?

**Contributions welcome**:
- Mathematical corrections
- Additional Pascal examples
- Alternative explanations
- Interview question additions
- Translation to other languages

**Not welcome**:
- Replacing Pascal with Python/NumPy (misses the pedagogical point)
- Adding framework dependencies (defeats the purpose)
- Obfuscating the implementations

---

## **Frequently Asked Questions**

### **Q: Why not just use PyTorch/TensorFlow documentation?**

**A**: Framework docs explain **how to use the API**, not **how it works internally**. This resource teaches the fundamentals that frameworks abstract away.

### **Q: Is this enough to get a job?**

**A**: Knowing these 18 equations won't get you hired alone, but **not knowing them will disqualify you** from serious ML roles. This is necessary but not sufficient.

### **Q: Should I memorize all 18?**

**A**: Memorize 1-7 (foundations + core attention). Understand 8-15. Be familiar with 16-18. You won't derive RoPE on a whiteboard, but you should know what it does.

### **Q: Where's the deep learning for computer vision content?**

**A**: See [GlassBoxAI-CNN](https://github.com/matthewJamesAbbott/GlassBoxAI-CNN) for convolutional architectures, pooling, batch normalization, and more.

### **Q: Why is this free?**

**A**: Knowledge should be accessible. The kid in Mumbai with a phone deserves the same resources as the Stanford grad student with a research cluster. We all deserve a fair go.

---

## **Acknowledgments**

This resource draws from:
- **"Attention is All You Need"** (Vaswani et al., 2017) - Transformer architecture
- **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022) - Chinchilla scaling
- **"LoRA: Low-Rank Adaptation of Large Language Models"** (Hu et al., 2021) - Parameter-efficient fine-tuning
- **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (Su et al., 2021) - RoPE mechanics

And countless Stack Overflow answers, blog posts, and whiteboard sessions with colleagues.

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
GitHub: [@matthewJamesAbbott](https://github.com/matthewJamesAbbott)

---

**Live Demo**: [https://glassbox-ai-core-math.vercel.app/](https://glassbox-ai-core-math.vercel.app/)

---

*"The only magic is the act we do that we don't understand. We make glassboxes here. No magic is allowed."*

*Built with precision. Explained with clarity. Understood completely.*
