# GenAI Defense Speaker Script

This script is written as spoken text, not as slide notes. The split keeps Ivan focused on the main GenAI decoding experiment and Aleksandr focused on robustness, ExAI audit, and interpretation.

## Speaker Split

| Slides | Speaker | Topic |
| --- | --- | --- |
| 1-12 | Ivan | Project idea, GenAI framing, theory, design, implementation, greedy baseline |
| 13-20 | Aleksandr | Gap analysis, decoding-grid result, masking and anti-repetition ablations |
| 21-23 | Aleksandr | ExAI/LRP audit layer and its limits |
| 24 | Ivan | Final GenAI results |
| 25-26 | Ivan + Aleksandr | Team contributions and conclusion |
| 27-28 | Ivan + Aleksandr | References and questions |

---

## Slide 1: Title

**Ivan**

Hello, we are Ivan Chabanov and Aleksandr Michailov, and our project is called "Decoding Amplifies Bias: Measuring Regard Under GPT-2 Decoding Choices."

The main idea is simple: we do not compare different language models. Instead, we take one fixed pretrained GPT-2 checkpoint and ask whether changing only the decoding strategy changes the social bias measured in generated text.

---

## Slide 2: Project Idea

**Ivan**

Our project studies open-ended text generation. We generate continuations from GPT-2 using several decoding strategies: greedy decoding, temperature sampling, top-k sampling, top-p sampling, and an optional no-repeat 3-gram constraint.

Then we score the generated text using regard labels. Regard means whether the text portrays a demographic group negatively, neutrally, positively, or as something outside those categories.

So the core question is: if the model is the same, the prompts are the same, and the generation length is the same, can decoding alone change measured social bias?

---

## Slide 3: GenAI Framing

**Ivan**

This is the main GenAI framing of the project. We treat decoding as the independent variable.

The model checkpoint is fixed: GPT-2 small. The prompt bank is fixed. The demographic groups are fixed. We also keep the seeds, sample count, and maximum number of generated tokens fixed.

That control is important because otherwise we could not say whether a change in bias comes from decoding, from a different model, from a different prompt set, or from a different sampling budget.

In our setup, the only thing we intentionally vary is the decoding algorithm.

---

## Slide 4: Relevant Theory - Decoding

**Ivan**

Autoregressive language models generate text one token at a time. At each step, GPT-2 gives us a probability distribution over the next token.

Decoding is the procedure that turns this distribution into an actual token sequence.

Greedy decoding always picks the most probable next token. Temperature sampling changes how sharp or flat the probability distribution is. Top-k sampling restricts sampling to the k most probable tokens. Top-p, or nucleus sampling, samples from the smallest group of tokens whose total probability mass passes a threshold. Finally, no-repeat n-gram decoding blocks local repetition.

These choices are often discussed as quality controls, but in this project we test whether they also affect measured bias.

---

## Slide 5: Relevant Theory - Why Decoding May Affect Bias

**Ivan**

The reason decoding can matter is that bias is not only inside the model weights. It is also reflected in which continuations are actually selected from the model distribution.

Greedy decoding can collapse into repetitive high-probability patterns. Sampling can produce more diverse and sometimes less repetitive text. Top-k and top-p restrict randomness in different ways, so they can produce different types of continuations.

If the generated text changes, the regard labels can also change. That means decoding can affect not only fluency and diversity, but also the measured social bias of the generated corpus.

---

## Slide 6: Relevant Theory - Regard

**Ivan**

For bias measurement, we use the regard framing from Sheng et al. Regard asks how a generated text portrays a demographic group.

The labels are negative, neutral, positive, and other. We mainly analyze the probability of negative regard for each group.

Our central gap metric is delta negative. It is the difference between the negative-regard rate for group A and group B under the same prompt type and decoding setting.

For example, if one group receives negative labels more often than another group for description prompts, that creates a positive negative-regard gap.

---

## Slide 7: Experimental Design

**Ivan**

Here is the controlled study configuration.

We use pretrained GPT-2 with no generator fine-tuning. The prompt bank has 12 templates, each instantiated with four demographic variants: Black woman, Black man, White woman, and White man.

We use four prompt types: occupation, description, aspiration, and achievement. For each setting we generate 50 samples per prompt and seed, with three seeds. The maximum generation length is 40 new tokens.

Across the 10 decoding configurations, this gives us 72,000 scored generations.

---

## Slide 8: Implementation

**Ivan**

The pipeline has five main stages.

First, we start from the fixed prompt bank. Second, GPT-2 generates continuations under a chosen decoding config. Third, we cache the generated continuations so reruns do not recompute the same outputs. Fourth, before scoring, we mask demographic mentions with XYZ. Fifth, the released regard classifier assigns labels, and we compute aggregate bias and quality metrics.

Besides regard gaps, we also compute quality controls: distinct-1, distinct-2, repeated 3-gram rate, and longest repetition span. This matters because a change in bias may be related to a change in degeneration or diversity.

---

## Slide 9: Pipeline Diagram

**Ivan**

This diagram summarizes the full system. The upper path is the main GenAI experiment: prompts go into GPT-2 decoding, we obtain generated text, score it with the regard classifier, and compute aggregate bias metrics.

The lower branch is the ExAI audit layer, which Aleksandr will discuss later. It is not the main generator experiment, but it helps inspect how the regard scorer behaves on individual generated examples.

---

## Slide 10: Reproducibility

**Ivan**

A major part of the implementation was reproducibility.

Generation outputs are cached. Manifests record the decoding config, seeds, prompt-bank digest, model name, sample count, and environment information. Scoring and metrics are stored separately from raw generations.

This is also important ethically. Generated text can contain offensive content, so the report and presentation use aggregate results and avoid publishing large raw dumps.

The main code is organized around generation, scoring, metrics, and quality modules.

---

## Slide 11: Results - Greedy Baseline

**Ivan**

The first result is the greedy baseline.

Greedy decoding is highly repetitive, but it already shows visible regard differences. In this table, Black woman and Black man have negative-regard rates of 0.417. White woman has 0.250, and White man has 0.167.

This does not prove a universal claim about all prompts or all models, but it gives a clear baseline pattern for our controlled prompt bank.

Greedy is also the most degenerate generation mode, so later we compare it with sampling strategies.

---

## Slide 12: Baseline Distribution Plot

**Ivan**

This plot gives the same baseline picture visually. It shows the regard distribution by demographic group under the baseline scored run.

The key point is that the groups are not receiving the same label distribution. Negative and positive regard are distributed differently across demographic variants, even though the prompt structure and model checkpoint are fixed.

Now Aleksandr will continue with the gap analysis and the full decoding grid.

---

## Slide 13: Results - Negative-Regard Gaps

**Aleksandr**

Thanks. After the baseline distribution, we compute pairwise negative-regard gaps within prompt types.

The table shows several representative greedy gaps. For occupation prompts, Black woman minus White woman has a gap of 0.250, and Black man minus White man also has 0.250. For description prompts, Black man minus White man reaches 0.500.

For the rest of the analysis, we highlight one trace: description prompts, Black man versus White woman. We use this trace because it is clearly visible in the baseline and can be followed across all decoding settings and ablations.

---

## Slide 14: Negative-Gaps Plot

**Aleksandr**

This plot shows the baseline negative-regard gaps. The important thing is not only the existence of a global difference, but that the differences depend on prompt type and group pair.

That is why the project reports per-prompt-type gaps instead of only one global aggregate. If we only reported a single overall number, we would hide where the effect is strongest.

---

## Slide 15: Results - Decoding Grid

**Aleksandr**

Now we move to the main GenAI result: the decoding grid.

Sampling clearly improves generation quality compared with greedy decoding. Greedy has distinct-2 around 0.002 and repeated 3-gram rate around 0.997, which is almost complete degeneration.

Sampling improves this a lot. Temperature 1.3 has distinct-2 around 0.411 and repeated 3-gram rate around 0.303. Top-k and top-p also improve diversity and reduce repetition.

But the key regard gap does not disappear. It changes in size, but it remains positive across all decoding settings shown here.

---

## Slide 16: Main GenAI Finding

**Aleksandr**

The main finding is that decoding is not only a quality parameter.

It strongly affects generation quality: greedy is repetitive, while sampling is much more diverse. But decoding also changes measured regard gaps. The gap can shrink or grow depending on the decoding strategy.

At the same time, sampling does not simply fix the bias pattern. In our highlighted description trace, the negative-regard gap remains positive for every decoding configuration.

So the conclusion is not "sampling solves bias." The conclusion is more careful: decoding changes both quality and measured bias, and those effects should be evaluated together.

---

## Slide 17: Week 5 Ablation - Masking

**Aleksandr**

Next, we tested whether our demographic masking step created the main result.

The standard regard-scoring workflow masks demographic mentions with XYZ before classification. This is meant to reduce direct reliance on identity words during scoring.

We compared masked and unmasked scoring across 240 prompt-type and group-pair rows. There were only two sign flips anywhere in the full comparison, and the largest absolute gap change was 0.020.

For the highlighted description trace, the gap stayed positive under every decoding configuration, both with masked and unmasked scoring.

---

## Slide 18: Week 5 Ablation - Anti-Repetition

**Aleksandr**

We also tested no-repeat 3-gram decoding.

This mostly helps quality. Distinct-2 improved in all 10 decoding configurations. Repetition generally went down.

But again, the main bias conclusion did not change. The highlighted gap stayed positive in all 10 configurations. Some values moved up or down, but there was no sign flip.

This tells us that repetition control improves the generated text, but it does not eliminate the measured regard asymmetry in this setup.

---

## Slide 19: Anti-Repetition Quality Plot

**Aleksandr**

This plot shows the quality effect of anti-repetition.

The important reading is that anti-repetition mainly behaves as expected: it improves diversity and reduces repeated 3-gram behavior in most configurations.

This supports the quality side of the experiment. It means we are not only measuring bias in obviously broken greedy text; we also test more reasonable generation settings.

---

## Slide 20: Anti-Repetition Gap Plot

**Aleksandr**

This plot shows how the regard gaps change under anti-repetition.

There are movements, but the highlighted trace does not switch sign. That is the robustness point: changing repetition control affects the outputs and metrics, but it does not overturn the main conclusion.

---

## Slide 21: Where ExAI Fits

**Aleksandr**

Now I will explain how the ExAI part fits into this GenAI project.

The LRP work is not an explanation of GPT-2 generation itself. We should be precise about that.

What it explains is the BERT-style regard classifier that scores GPT-2 outputs. So it is an audit layer for the evaluation pipeline.

This matters because our bias conclusions depend on automatic scoring. LRP helps inspect whether the scorer is reacting to meaningful content words, demographic tokens, repetition, punctuation, or other artifacts.

---

## Slide 22: ExAI Audit Results

**Aleksandr**

The ExAI audit was useful, but we present it with limits.

The local BERT regard classifier had 0.639 held-out accuracy and 0.483 macro F1. Agreement with the released scorer was 0.694. On 12 generated audit examples, only 3 agreed with the scoring labels, which suggests domain shift between ordinary regard data and GPT-2 generated continuations.

The heatmaps surfaced cases where identity tokens had relevance for a negative-regard class. That is useful as an audit question.

But faithfulness was mixed: removing top-attributed tokens did not reduce the target score more than random removal. So the ExAI layer improves transparency, but it is not causal proof.

---

## Slide 23: Token-Relevance Heatmap

**Aleksandr**

This is an example of the token-level explanation output.

The goal is not to claim that one heatmap proves bias. The goal is to make individual scoring decisions inspectable.

For example, if identity-bearing tokens receive strong relevance for a negative class, then we know this example deserves closer inspection. This complements the aggregate GenAI metrics, but it does not replace them.

---

## Slide 24: Final Results

**Ivan**

To summarize the GenAI contribution, we implemented a controlled GPT-2 decoding study with 72,000 scored generations.

The full grid shows that sampling strongly improves quality over greedy decoding. It also changes measured regard gaps, so decoding is part of the bias-measurement story.

However, the highlighted description gap did not disappear under temperature, top-k, top-p, masking sensitivity, or anti-repetition.

So our final claim is careful: decoding affects measured social bias, but better decoding does not automatically remove it.

---

## Slide 25: Team Contributions

**Ivan**

For the team split, I worked mainly on the GenAI side: the GPT-2 generation pipeline, prompt bank, caching and manifests, decoding grid, regard scoring, metrics, plots, and final GenAI analysis.

**Aleksandr**

I worked mainly on the ExAI side: the BERT regard classifier, LRP implementation, heatmap rendering, faithfulness and sensitivity validation, and the audit benchmark.

**Ivan**

Both of us contributed to writing, interpretation, and preparing the final presentation.

---

## Slide 26: Conclusion

**Aleksandr**

The main conclusion is that decoding is not only a fluency or diversity choice. It changes the distribution of generated text, and that changes measured social bias.

**Ivan**

Sampling improves quality and sometimes reduces gaps, but it does not eliminate the highlighted negative-regard gap in our controlled setup. The ExAI layer then helps us audit the scorer behind those measurements.

**Aleksandr**

So the project combines a GenAI experiment with an explainability check: first we measure how decoding affects generated text and regard bias, and then we inspect whether the automatic scorer behind those labels is trustworthy enough to interpret.

---

## Slide 27: References

**Ivan**

The main references are Sheng et al. for regard and bias in language generation, Holtzman et al. for neural text degeneration and decoding, GPT-2 as the generator, BERT as the classifier architecture, and LRP for the ExAI audit layer.

---

## Slide 28: Questions

**Ivan**

Thank you. We are ready for questions.

**Aleksandr**

And we can answer both parts: the main GenAI decoding experiment and the ExAI audit layer.
