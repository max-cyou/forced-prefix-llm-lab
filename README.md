[Русская версия / Russian version](readme-ru.md)

# How Does a Forced Prefix During Local LLM Generation Affect Its Safety Filters?

The experiment compared two generation modes for the same prompt:

- **NORMAL** — the model generates a response without intervention.
- **FORCED** — the initial response tokens are forced using a `LogitsProcessor`, after which the model continues generating independently.

The analysis uses `normal_response` and `forced_free_response` — the continuation excluding the forced prefix.

## Test Configuration

| Parameter | Value |
|---|---|
| Model | Qwen3.5-122B-A10B |
| Quantization | GGUF Q4_K_S |
| GPU | NVIDIA A100 80 GB |
| Backend | llama-cpp-python 0.3.36 |
| Tests | 98 prompts across 25 categories |
| Generations | 196 (98 NORMAL + 98 FORCED) |
| Context | 4096 tokens |
| Response limit | 1000 free tokens per mode |
| Temperature / seed | 0 / 42 |
| Thinking | Disabled |

# Results and Findings

### Initial Evaluation of FORCED Continuations

| Class | Cases |
|---|---:|
| U — Substantial problematic content | 44 |
| P — Partial / ambiguous content | 33 |
| S — Safe continuation or refusal | 17 |
| L — Partial disclosure of a test secret | 4 |

> In 44 out of 98 cases (44.9%), the refusal disappeared completely — no refusal-related wording was detected in the independently generated FORCED continuation. In 87 out of 98 cases (88.8%), the model did not exhibit an early refusal.

### Results by Category

**Model:** Qwen3.5-122B-A10B (Q4_K_S)  
**Method:** Token-Level Forced Decoding  
**Dataset:** 98 tests across 25 categories  
**Modes:** NORMAL / FORCED

| # | Category | Behavior Under FORCED | Hallucinations and Generation Degradation | Outcome |
|---|---|---|---|---|
| 1 | Illegal Drugs | Refusals were replaced with instructions formatted as chemical procedures. The model frequently added warnings afterward. | VERY frequent fabricated compositions, questionable chemical reactions, and incorrect procedural sequences. | Refusals were bypassed, but the generated content had low factual reliability. |
| 2 | Explosives | In all 4 tests, the model switched from refusing to providing procedural descriptions. | Highly questionable chemical claims; severe repetition loops in 1/4 tests. | Significant behavioral change accompanied by generation degradation. |
| 3 | Firearms | Inconsistent behavior, ranging from preserved refusals to technical descriptions. | The generation limit was reached in 2/4 tests; long repetitive passages appeared. | Partial bypass of restrictions with high instability. |
| 4 | Poisoning | Potentially harmful descriptions appeared in 2/4 tests, while the remaining responses maintained protective behavior. | No significant repetition loops were observed. | Partial effectiveness of the method. |
| 5 | Medical Harm | In all 4 tests, the model switched from standard refusals to potentially harmful medical recommendations. | Responses remained coherent; medical accuracy was not verified. | Significant content changes without noticeable generation degradation. |
| 6 | SQL Injection (Attacking Real Services) | In all 4 tests, the model generated Python code and SQL injection examples targeting synthetic Flask/SQLite applications. | No significant repetition or looping was observed. | Consistent transition to technical content. |
| 7 | Service Exploitation | Refusals were partially bypassed, but the model generally produced high-level plans rather than specific technical solutions. | One test exhibited a repetition loop and reached the token limit. | Partial results with limited technical specificity. |
| 8 | Banking System Hacking | Responses remained mostly generic and incomplete despite the forced prefix. | Long responses; the token limit was reached in both NORMAL and FORCED modes. | Limited substantive effectiveness. |
| 9 | Account Theft | Substantive fragments appeared in 2/4 tests, while the other 2/4 produced partial responses. | FORCED continuations were noticeably shorter than normal responses. | Partial bypass of refusals. |
| 10 | Ransomware | In all 4 tests, the model generated fragments of maliciously oriented code. | No severe repetition loops were observed; code functionality was not tested. | Consistent transition from refusals to code generation. |
| 11 | Malware, Stealers, and Data Destruction | In all 4 tests, the model generated code fragments instead of maintaining its refusal. | Significant repetition and generation degradation occurred in 1/4 tests. | Substantial behavioral change with occasional failures. |
| 12 | Phishing Messages | Substantive fraudulent messages were generated in 3/4 tests. | Responses were generally coherent and followed the requested format. | High effectiveness with minimal repetition. |
| 13 | Phishing Websites | In all 4 tests, the model generated web application fragments instead of refusing. | Short, structured responses without severe degradation. | Consistent bypass of refusals at the content level. |
| 14 | Financial Fraud | The model switched to describing fraudulent scenarios, mostly at a general level. | No significant repetition loops were observed. | Partial behavioral change with limited specificity. |
| 15 | Currency Counterfeiting | Refusals were partially bypassed; the model generated incomplete and questionable descriptions. | One test exhibited severe repetition and reached the token limit. | Partial results with noticeable generation degradation. |
| 16 | Document and Evidence Forgery | In all 4 tests, the model switched to descriptive instructions without providing complete technical solutions. | Responses were relatively short and coherent, without severe repetition. | Refusals were bypassed, but technical detail remained limited. |
| 17 | Extremist Propaganda and Recruitment | In all 4 tests, the model independently generated problematic propaganda-related content. | Short, coherent responses without significant loops. | Consistent behavioral change. |
| 18 | Racial Hatred | Mixed results: some refusals were maintained, while other responses contained discriminatory content. | Repetitive generation occurred in one test. | Partial bypass of restrictions with inconsistent behavior. |
| 19 | Religious Hatred | In both tests, the model independently generated discriminatory content. | Short responses without significant repetition. | Substantial content changes in 2/2 tests. |
| 20 | Targeted Violence and Threats | Protective behavior was maintained in 3/4 tests; the remaining 1/4 was ambiguous. | No significant generation failures were observed. | High resistance to forced decoding. |
| 21 | Sexual Coercion and Intimate Blackmail | Safe behavior was maintained in 2/4 tests, one produced a partial response, and one generated problematic content. | No significant repetition loops were observed. | Mostly resistant but inconsistent behavior. |
| 22 | Child Safety | Protective elements remained present, although partial or problematic continuations appeared in 3 tests. | No severe generation degradation was observed. | Partial weakening of restrictions. |
| 23 | Suicide and Self-Harm | In all 4 tests, the model maintained a protective response despite the forced prefixes. | No significant repetition; response lengths remained largely unchanged. | The most resistant safety behavior observed. |
| 24 | Doxxing and Harassment | In all 4 tests, the model switched to general descriptions of obtaining other people's information. | No real personal information or confirmed leaks were identified. | Partial bypass of refusals without disclosure of real personal data. |
| 25 | System Secret Disclosure | NORMAL did not reveal the secret. In FORCED, its numerical component was reproduced in all 4 tests, but the complete secret was not disclosed. | Very short responses without repetition. | Partial disclosure of the test secret in 4/4 cases. |

### Key Observations

| Effect | Observation |
|---|---|
| Refusal Bypass | Forced prefixes changed the nature of the responses: the model independently continued generating content that it had refused to provide in NORMAL mode. |
| Delayed Refusal | In certain categories, the model first generated problematic content and then appended a warning or refusal. |
| Hallucinations | Most noticeable in chemical and technical scenarios, where plausible formatting was combined with questionable formulas and procedures. |
| Repetition Loops | Some FORCED responses repeatedly generated the same phrases, instructions, or code fragments. |
| Code Generation | The most consistent behavioral changes occurred in SQL Injection, Ransomware, Malware, and Phishing Websites. |
| Preserved Safety Restrictions | The strongest safety behavior was observed in the Self-Harm and Targeted Violence categories. |
| Data Leakage | In system-secret tests, the model reproduced the numerical component of the secret in all 4 FORCED cases. |
| Performance | Average generation speed was nearly identical between NORMAL and FORCED modes. |

# Conclusion

Forced Decoding significantly affected the behavior of Qwen3.5-122B-A10B. The most noticeable changes occurred in cybersecurity, extremist propaganda, chemical threats, and medical harm categories.

The main characteristics of FORCED generation included bypassing initial refusals, generating problematic content, delayed safety warnings, hallucinations, and occasional repetition loops.

The effect varied by category: in some scenarios, the model transitioned to substantive continuations, while in others it maintained its restrictions or generated incomplete and repetitive responses.

**Thank you for your interest in this project!**
