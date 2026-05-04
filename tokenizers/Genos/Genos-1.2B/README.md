---
license: mit
tags:
- biology
---

<div align="center">
  <img src="https://raw.githubusercontent.com/BGI-HangzhouAI/Genos/main/images/Genos_model.png" width="100%" />
</div>

# Genos

Genos, as a foundational model in the field of human genomics, trained on hundreds of high-quality genome reference data, has achieved the ability to contextually model human genome sequences up to millions of base pairs. Through single-base resolution learning, this model possesses the capability to identify hidden deep sequence patterns and functional features within genomes, providing scientists with a new research method that connects genetic information with life activities.

For instructions, details, and examples, please refer to the [Genos GitHub](https://github.com/BGI-HangzhouAI/Genos).

Below are the data volume of our model training and related parameters.

<table align="center">
  <tr>
    <th>Model Specification</th>
    <th>Genos 1.2B</th>
    <th>Genos 10B</th>
  </tr>

  <!-- Model Scale category title - span 3 columns -->
  <tr>
    <td colspan="3" align="center"><b>Model Scale</b></td>
  </tr>
  <tr>
    <td>Total Parameters</td>
    <td>1.2B</td>
    <td>10B</td>
  </tr>
  <tr>
    <td>Activated Parameters</td>
    <td>0.33B</td>
    <td>2.87B</td>
  </tr>
  <tr>
    <td>Trained Tokens</td>
    <td>1600 B</td>
    <td>2200 B</td>
  </tr>

  <!-- Architecture category title - span 3 columns -->
  <tr>
    <td colspan="3" align="center"><b>Architecture</b></td>
  </tr>
  <tr>
    <td>Architecture Type</td>
    <td>MoE</td>
    <td>MoE</td>
  </tr>
  <tr>
    <td>Number of Experts</td>
    <td>8</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Selected Experts per Token</td>
    <td>2</td>
    <td>2</td>
  </tr>
  <tr>
    <td>Number of Layers</td>
    <td>12</td>
    <td>12</td>
  </tr>
  <tr>
    <td>Attention Hidden Dimension</td>
    <td>1024</td>
    <td>4096</td>
  </tr>
  <tr>
    <td>Number of Attention Heads</td>
    <td>16</td>
    <td>16</td>
  </tr>
  <tr>
    <td>MoE Hidden Dimension (per Expert)</td>
    <td>4096</td>
    <td>8192</td>
  </tr>
  <tr>
    <td>Vocabulary Size</td>
    <td>128 (padded)</td>
    <td>256 (padded)</td>
  </tr>
  <tr>
    <td>Context Length</td>
    <td>up to 1M</td>
    <td>up to 1M</td>
  </tr>
</table>


Genos 1.2B and 10B checkpoints are available here:

- [Genos-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-1.2B)
- [Genos-10B](https://huggingface.co/BGI-HangzhouAI/Genos-10B) 

We also provide checkpoints trained under the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) framework:

- [Genos-Megatron-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-1.2B) 
- [Genos-Megatron-10B](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-10B)
