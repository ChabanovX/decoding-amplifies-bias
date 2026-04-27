# GenAI Defense Speaker Script, 5-Minute Version

Target timing: about 5 minutes total. Speak briskly and advance image-only/reference slides quickly.

## Speaker Split

| Slides | Speaker | Timing | Topic |
| --- | --- | ---: | --- |
| 1-12 | Ivan | ~2:30 | Project idea, GenAI setup, theory, implementation, baseline |
| 13-23 | Aleksandr | ~2:00 | Decoding-grid results, ablations, ExAI audit |
| 24-28 | Ivan + Aleksandr | ~0:30 | Final claim, contributions, conclusion, questions |

---

## Slides 1-2: Title and Project Idea

**Ivan**

Hello, we are Ivan Chabanov and Aleksandr Michailov. Our project is "Decoding Amplifies Bias."

The question is: if GPT-2 is fixed, can decoding alone change measured social bias in generated text?

We generate continuations with greedy, temperature, top-k, top-p, and no-repeat 3-gram decoding. Then we score the generations with regard labels: negative, neutral, positive, or other.

So this is a GenAI experiment about how decoding affects output quality and measured bias.

---

## Slides 3-6: GenAI Framing and Theory

**Ivan**

The key design choice is control. We do not compare different models. We fix GPT-2 small, prompts, demographic groups, seeds, sample count, and max length. The variable we change is decoding.

Autoregressive models generate one token at a time. Decoding decides how we choose each next token.

Greedy chooses the most likely token. Temperature changes distribution sharpness. Top-k samples from the k most likely tokens. Top-p samples from the smallest token set whose probability mass reaches p.

These are usually quality settings, but they can also change which portrayals appear in generated text.

For bias measurement we use regard: how positively or negatively text portrays a demographic group. Our main metric is the negative-regard gap between two groups under the same prompt type.

---

## Slides 7-10: Experimental Design and Implementation

**Ivan**

We use 12 prompt templates, four demographic variants, four prompt types, three seeds, and 50 samples per prompt per seed. Across 10 decoding configs, this gives 72,000 scored generations.

The pipeline is: prompt bank, GPT-2 generation, cached continuations, demographic masking with `XYZ`, regard scoring, then bias and quality metrics.

We track quality with distinct-1, distinct-2, repeated 3-gram rate, and longest repetition span.

For reproducibility, generations are cached and manifests record seeds, configs, prompt digest, and environment. We avoid large raw dumps because outputs may contain offensive content.

---

## Slides 11-12: Greedy Baseline

**Ivan**

The greedy baseline shows two things.

First, greedy decoding is highly repetitive. Second, regard distributions differ across groups. In the baseline table, Black woman and Black man have higher negative-regard rates than White woman and White man.

This motivates the full grid: does sampling improve quality, change gaps, or remove them?

Aleksandr will continue with those results.

---

## Slides 13-16: Decoding-Grid Results

**Aleksandr**

The main result is the decoding-grid slide.

Sampling clearly improves quality. Greedy has almost no diversity: distinct-2 is 0.002, and repeated 3-gram rate is 0.997. With sampling, diversity rises and repetition falls. Temperature 1.3 reaches distinct-2 of 0.411.

But the key bias gap does not disappear. We track description prompts for Black man versus White woman. Greedy has gap 0.250. Temperature 1.3 reduces it to 0.133, but the gap stays positive across the grid.

So decoding changes quality and measured bias, but better decoding does not automatically remove the bias pattern.

---

## Slide 17: Masking Ablation

**Aleksandr**

Slide 17 checks whether demographic masking created the result.

We replace demographic mentions with `XYZ` before scoring. To test this, we scored the same generations twice: masked and unmasked.

Across 240 comparisons, only two changed sign, and the largest gap change was 0.020. For the main Black man versus White woman trace, the gap stayed positive in every decoding configuration.

So the main result is not caused by `XYZ` masking.

---

## Slides 18-20: Anti-Repetition Ablation

**Aleksandr**

Next we tested no-repeat 3-gram decoding.

It improves quality: distinct-2 improves in all 10 configs, and repetition usually decreases. But it does not change the bias conclusion.

The highlighted gap stays positive in all configs. Anti-repetition helps degeneration, but it does not remove the measured regard asymmetry.

---

## Slides 21-23: ExAI Audit Layer

**Aleksandr**

The ExAI part is useful, but we frame it carefully.

LRP does not explain GPT-2 generation directly. It explains the BERT-style regard classifier that scores GPT-2 outputs. So it audits the evaluation pipeline.

This matters because our bias metrics depend on automatic scoring. LRP heatmaps show whether the scorer reacts to content words, demographic tokens, repetition, or punctuation artifacts.

The audit found warning signs, including identity tokens relevant for a negative class. But faithfulness was mixed, so ExAI is transparency evidence, not causal proof.

---

## Slide 24: Final Results

**Ivan**

Our final claim is that decoding is not just a fluency knob.

In this controlled GPT-2 study, decoding changes generated text, quality metrics, and regard gaps. However, the highlighted negative-regard gap remains positive under the full grid and both ablations.

---

## Slide 25: Team Contributions

**Ivan**

I worked mainly on the GenAI pipeline: GPT-2 generation, prompt bank, caching, decoding grid, regard scoring, metrics, plots, and analysis.

**Aleksandr**

I worked mainly on the ExAI part: BERT regard classifier, LRP explanations, heatmaps, validation checks, and audit benchmark.

Both of us worked on interpretation and final materials.

---

## Slides 26-28: Conclusion and Questions

**Aleksandr**

To conclude: sampling improves generation quality, but it does not automatically solve measured bias.

**Ivan**

The GenAI contribution is the controlled decoding study. The ExAI contribution is the scorer audit that helps us inspect the evaluation layer.

Thank you. We are ready for questions.
