---
marp: true
theme: uncover
class: invert
paginate: true
style: |
  section { font-size: 24px; }
  table { font-size: 20px; }
  .small { font-size: 18px; }
---

<!-- _class: lead invert -->

# **Decoding Amplifies Bias**

### Measuring Regard Under GPT-2 Decoding Choices

Ivan Chabanov & Aleksandr Michailov

---

## Project Idea

**Question:** if the language model checkpoint is fixed, can decoding alone change measured social bias in generated text?

We study GPT-2 open-ended generation under:

- greedy decoding
- temperature sampling: 0.7, 1.0, 1.3
- top-k sampling: 20, 50, 100
- top-p sampling: 0.8, 0.9, 0.95
- optional no-repeat 3-gram decoding

Bias is measured with **regard**: whether a generation portrays a demographic negatively, neutrally, positively, or as other.

---

## GenAI Framing

This is a **controlled generation experiment**, not a model-comparison benchmark.

We keep fixed:

- GPT-2 small checkpoint
- prompt bank
- demographics
- seeds
- sample count
- maximum generation length

Then we vary only the decoding algorithm and measure how the output distribution changes.

---

## Relevant Theory: Decoding

Autoregressive language models generate one token at a time:

$$p(x_t \mid x_{<t})$$

Decoding turns that probability distribution into text.

| Method | Effect |
| --- | --- |
| Greedy | always picks the highest-probability token |
| Temperature | sharpens or flattens token probabilities |
| Top-k | samples from the k most likely tokens |
| Top-p | samples from the smallest set whose mass exceeds p |
| No-repeat n-gram | blocks repeated n-gram continuations |

---

## Relevant Theory: Why Decoding May Affect Bias

Decoding changes which continuations are likely to appear.

- Greedy can collapse into repetitive, high-probability patterns.
- Higher temperature increases diversity and lower-probability continuations.
- Top-k and top-p restrict randomness differently.
- Anti-repetition changes local text quality and may change classifier inputs.

So the same model can produce different **measured regard distributions** under different decoding choices.

---

## Relevant Theory: Regard

We use the framing from Sheng et al.:

> regard measures how positively or negatively generated text portrays a demographic group.

Labels:

```text
negative | neutral | positive | other
```

Primary metric:

$$\Delta_{neg} = P(negative \mid group A) - P(negative \mid group B)$$

We report group distributions and bootstrap confidence intervals.

---

## Experimental Design

| Item | Value |
| --- | --- |
| Generator | pretrained `gpt2`, no fine-tuning |
| Prompt bank | 12 templates x 4 demographic variants |
| Prompt types | occupation, description, aspiration, achievement |
| Demographics | Black woman, Black man, White woman, White man |
| Samples | 50 per prompt per seed |
| Seeds | 0, 1, 2 |
| Max new tokens | 40 |
| Decoding configs | 10 |
| Final scale | 72,000 scored generations |

---

## Implementation

Pipeline:

```text
fixed prompt bank
        |
GPT-2 generation under decoding config
        |
cached generated continuations
        |
demographic masking with XYZ
        |
released regard classifier
        |
bias metrics + quality metrics + bootstrap CIs
```

Quality controls:

- distinct-1 / distinct-2
- repeated 3-gram rate
- longest repetition span

---

![bg contain](images/main_pipeline.png)

---

## Reproducibility

The project was built around cached, rerunnable artifacts.

- generation cache avoids recomputation
- manifests record config, seeds, prompt-bank digest, environment
- scoring artifacts separate raw generations from aggregate reports
- outputs may contain offensive text, so the report avoids large raw dumps

---

## Results: Greedy Baseline

Greedy decoding produced strong repetition and visible regard gaps.

| Demographic | Neg | Neu | Pos | Other |
| --- | ---: | ---: | ---: | ---: |
| Black woman | 0.417 | 0.000 | 0.500 | 0.083 |
| Black man | 0.417 | 0.167 | 0.250 | 0.167 |
| White woman | 0.250 | 0.000 | 0.750 | 0.000 |
| White man | 0.167 | 0.083 | 0.750 | 0.000 |

Greedy was useful as a baseline, but it was also the most degenerate generation mode.

---

![bg contain](outputs/plots/fbe608112493c39dd4d4_regard_distribution.png)

---

## Results: Negative-Regard Gaps

Representative greedy gaps:

| Prompt type | Comparison | Gap |
| --- | --- | ---: |
| occupation | Black woman - White woman | 0.250 |
| occupation | Black man - White man | 0.250 |
| description | Black man - White woman | 0.250 |
| description | Black man - White man | 0.500 |

The main trace we follow later:

```text
description / Black man vs White woman
```

---

![bg contain](outputs/plots/fbe608112493c39dd4d4_negative_gaps.png)

---

## Results: Decoding Grid

Sampling improved diversity and reduced degeneration.

| Config | Distinct-2 | Repeated 3-gram | Key gap |
| --- | ---: | ---: | ---: |
| Greedy | 0.002 | 0.997 | 0.250 |
| Temperature 0.7 | 0.270 | 0.472 | 0.185 |
| Temperature 1.0 | 0.358 | 0.353 | 0.193 |
| Temperature 1.3 | 0.411 | 0.303 | 0.133 |
| Top-k 50 | 0.358 | 0.350 | 0.163 |
| Top-p 0.9 | 0.316 | 0.390 | 0.152 |

The key gap shrinks in some settings, but it stays positive.

---

## Main Finding

Decoding has a large effect on **generation quality**.

- Greedy: extremely repetitive, very low diversity.
- Sampling: much higher distinct-2, lower repeated 3-gram rate.
- Temperature 1.3 gives the highest diversity in our grid.

Decoding has a smaller but real effect on **measured regard bias**.

- Bias gaps move across decoding settings.
- The highlighted description gap stays positive for every setting.
- Sampling improves fluency/diversity, but does not erase the measured asymmetry.

---

## Masking

We tested whether demographic masking with `XYZ` created the main result.

Across 240 compared prompt-type/group-pair rows:

- sign flips: 2
- largest absolute gap change: 0.020
- key trace: positive under every decoding config

| Config | Masked gap | Unmasked gap |
| --- | ---: | ---: |
| Greedy | 0.250 | 0.250 |
| Temperature 1.3 | 0.133 | 0.137 |
| Top-k 50 | 0.163 | 0.162 |
| Top-p 0.9 | 0.152 | 0.147 |

---

## Anti-Repetition

No-repeat 3-gram mostly improves quality, but does not change the main conclusion.

| Config | Base gap | Anti-rep gap | Sign flip? |
| --- | ---: | ---: | --- |
| Greedy | 0.250 | 0.250 | no |
| Temperature 1.0 | 0.193 | 0.212 | no |
| Top-k 50 | 0.163 | 0.212 | no |
| Top-p 0.8 | 0.208 | 0.160 | no |
| Top-p 0.9 | 0.152 | 0.205 | no |

Distinct-2 improved in 10/10 configs; the key bias trace stayed positive in 10/10 configs.

---

![bg contain](outputs/plots/week5_antirep_quality_delta.png)

---

![bg contain](outputs/plots/week5_antirep_gap_delta.png)

---

## Explaining the Results

The LRP work is useful for the GenAI defense, but as an **evaluation audit layer**.

It does **not** explain GPT-2 generation directly.

It explains the BERT-style regard classifier used to score generated text:

```text
GPT-2 output -> regard classifier -> label
                              |
                              v
                    LRP token relevance
```

This helps us audit whether labels are driven by content words, demographic tokens, repetition, or punctuation artifacts.

---

## ExAI Audit Results

The audit layer found useful but limited evidence.

- Local BERT regard classifier: 0.639 held-out accuracy, 0.483 macro F1.
- Agreement with released scorer: 0.694 accuracy.
- On 12 generated audit examples: only 3/12 agreed with scoring labels.
- LRP heatmaps surfaced cases where identity tokens had relevance for negative regard.
- Faithfulness was mixed: top-token removal was not stronger than random removal.

Interpretation: ExAI increases transparency of the scoring layer, but it is not causal proof.

---

![bg contain](images/more_token_relev.png)

---

## Final Results

What we can defend for the GenAI course:

1. We implemented a controlled GPT-2 decoding study.
2. The full grid produced 72,000 scored generations.
3. Sampling strongly improved quality over greedy decoding.
4. Decoding changed measured regard gaps, but did not eliminate the highlighted gap.
5. Masking and anti-repetition ablations did not overturn the conclusion.
6. ExAI/LRP made the automatic scoring layer more inspectable.

---

## Team Contributions

| Team Member | Contributions |
| --- | --- |
| Ivan Chabanov | GPT-2 generation pipeline, prompt bank, caching/manifests, decoding grid, metrics, plots, heatmap rendering |
| Aleksandr Michailov | ExAI module, BERT regard classifier, LRP implementation, regard scoring, faithfulness/sensitivity validation, audit benchmark |

We both contributed to writing, interpretation, and final presentation materials.

---

## Conclusion

The project shows that decoding is not only a quality knob.

It changes the distribution of generated text, which changes measured social bias. Sampling reduces degeneration and sometimes reduces regard gaps, but the main highlighted negative-regard gap remains positive across all tested decoding settings.

The ExAI part strengthens the defense by auditing the scorer behind those measurements.

---

## Thank you for your time!

### References

- Sheng et al., "The Woman Worked as a Babysitter" - [ACL Anthology](https://aclanthology.org/D19-1339/)
- Holtzman et al., "The Curious Case of Neural Text Degeneration" - [arXiv](https://arxiv.org/abs/1904.09751)
- Radford et al., GPT-2
- Devlin et al., BERT - [arXiv](https://arxiv.org/abs/1810.04805)
- Bach et al., Layer-wise Relevance Propagation - [PLOS ONE](https://doi.org/10.1371/journal.pone.0130140)
- Released regard scorer - [`sasha/regardv3`](https://huggingface.co/sasha/regardv3)

