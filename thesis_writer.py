import json
import time
import pyperclip
from pyautogui import press, hotkey, sleep, typewrite, locateOnScreen
from tqdm import tqdm


def load_file(file: str) -> str:
    with open(file, 'r') as f:
        return f.read()


FOLDER = R'C:\Users\berti\OneDrive\Desktop\Masterarbeit schreiben'
DATA = 'MA komplett-Beschreibung.txt'

STRUKTUR = load_file(f'{FOLDER}\\Struktur.txt')

SECTIONS = [
    'Einleitung / Motivation',
    'Theoretischer Hintergrund',
    'Current State of Research',
    'Methodologie',
    'Entwicklung',
    'Evaluation',
    'Zusammenfassung und Ausblick',
]
SECTION_FILES = [f'{FOLDER}\\Gliederung\\0{section_number}.txt' for section_number in range(1, len(SECTIONS) + 1)]

GENERATING_ICON = R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\generating.png'


def open_browser(browser: str = 'chrome'):
    press('win')
    sleep(0.5)
    typewrite(browser)
    press('enter')
    sleep(2)


def new_tab():
    hotkey('ctrl', 't')
    sleep(0.5)
    typewrite('chatgpt.com')
    press('a')
    press('backspace')
    press('enter')
    sleep(5)


def query_chat(query: str):
    pyperclip.copy(query)
    sleep(0.1)
    hotkey('ctrl', 'v')
    press('enter')

    sleep(2)

    while True:
        try:
            locateOnScreen(GENERATING_ICON)
            sleep(1)
        except Exception:
            break

    sleep(2)


def chunked_file(file: str, delimiter: str = '\n\n\n') -> list[str]:
    # read a file and split it by a delimiter, removing empty lines
    data = load_file(file).split(delimiter)

    chunks = [line for line in data if line.strip(' \n')]
    assert chunks, f'No data found in {file}'
    return chunks


def get_additional_information(section_name: str, subsection: str) -> str | None:
    if section_name == SECTIONS[0]:
        return R"""Das folgende ist eine alte Version der Einleitung, die als Referenz für die Erstellung der neuen Einleitung dienen kann. Die Zitierungen und Referenzen können gerne übernommen werden, aber die Struktur und der Inhalt sollten an die spezifischen Anforderungen der Arbeit angepasst werden.

\section{Introduction and Motivation}

The capacity to identify and assign specific competencies is of critical importance in today's interconnected research and industrial landscape for effective collaboration \cite{rodrigues_competence_2004, coates_study_2007}. Highlighted by Rodrigues et al. two decades ago, this challenge has become more relevant than ever. There is a clear necessity to link highly specialized subject areas that could benefit from cooperation through the efficient allocation and visibility of competencies \cite{rodrigues_competence_2004, meyer_employee_2015}. The main issue lies in the fact that many of these potential collaborations remain unrealized because the involved parties—from individual researchers to large firms—cannot find each other \cite{gurcan_extraction_2019, meyer_employee_2015, coates_study_2007}. This is often due to competencies not being clearly defined or made visible, preventing effective networking \cite{gurcan_extraction_2019, coates_study_2007}.

Twenty years after Rodrigues et al., this thesis re-examines this problem at a time when advancements in \gls{AI}, particularly in the field of \gls{NLP}, offer new solutions.

This thesis is primarily focused on how innovations in \gls{AI} and \gls{NLP} can be utilized to enhance the extraction of competencies. Competence extraction finds critical applications across multiple settings, serving not only as a tool for networking within the academic realm but also in diverse business environments. For example, in academia, this technology could streamline the formation of interdisciplinary research teams by accurately matching researchers based on their skills and knowledge areas. In the corporate sector, it can improve human resource strategies by facilitating the discovery and alignment of employee capabilities with strategic business needs.

Building on these examples, the objective extends to the development of a generic system capable of operating across various domains. This approach will create a foundation for diverse solution approaches, enabling the application of competence extraction in other contexts.

A significant area that has already been extensively researched is the extraction of competencies from job postings \cite{li_skillgpt_2023, decorte_extreme_2023, nguyen_rethinking_2024, magron_jobskape_2024}. This research builds upon these insights, aiming to develop a system capable of handling a broader array of document types. This includes processing extensive and complex texts, such as multiple publications from the same author, to extract a unified set of competencies. The challenge lies in adapting to the variety of formats and complexity of these documents, which requires innovative, domain-agnostic solutions that go beyond the relatively structured environments of job postings.




\subsection{Relevance to the Kompetenzpool Research Project}

The \textit{Kompetenzpool}\footnote{https://www.for.kit.edu/kompetenznetzwerk.php} research project, conducted by the Karlsruhe Institute of Technology (KIT), focuses on optimizing the assignment of scientific personnel to relevant tasks and collaborations. The Kompetenzpool serves as a specialized social network designed to connect scientific staff by making their competencies visible to others. This network facilitates the enhancement of collaboration among researchers from diverse domains by allowing them to identify colleagues with the necessary expertise for their projects.

In essence, the Kompetenzpool enables researchers to find and collaborate with colleagues whose skills complement their needs, thereby fostering interdisciplinary research and improving overall research efficiency. To achieve this, the system must accurately represent the competencies of each researcher. Currently, efforts are underway to automate the suggestion of competencies for each scientific staff member based on their publications and other academic outputs.

While existing research has made significant strides in this area, the current state of competency extraction is still inadequate for the precision required by the Kompetenzpool. The primary goal of this thesis is to enhance the competency extraction process, resulting in more accurate identification of researchers' skills. This improvement is expected to significantly bolster the functionality of the Kompetenzpool, thereby supporting the broader goals of the research project.

For a detailed review of existing research in this area, see \todo{Insert references to relevant research here}.


\subsection{Research Objectives} 

\begin{itemize}
    \item Develop a generalized system for competency extraction.
    \item Conduct a rigorous evaluation to assess the impact of various methodologies on the quality of the competencies extracted.
    \item Utilize expert evaluations to validate the findings.
    \item Analyze domain agnostic usability of the system.
    \item Develop fine tuning system to optimize a LLM for competence extraction.
\end{itemize}

\subsection{Structure of the Thesis}

This thesis is structured as follows:

\begin{itemize}
    \item \textbf{Theoretical Background:} This chapter provides an in-depth review of the theoretical concepts essential to the thesis. It covers competency extraction, natural language processing (NLP), large language models (LLMs), retrieval-augmented generation (RAG), structured output, and various fine-tuning techniques, including reinforcement learning from human feedback (RLHF).
    
    \item \textbf{State of Current Research:} This chapter examines the latest advancements in artificial intelligence (AI) and NLP, particularly in the context of competency extraction from job postings. It also reviews recent developments in reinforcement learning from human feedback.
    
    \item \textbf{Methodology:} This chapter details the methods used in this research. It includes descriptions of the data, extraction processes, evaluation techniques, and the domain-agnostic approach to competency extraction. Additionally, it outlines the iterative model refinement process.
    
    \item \textbf{Development of the Competence Extraction System:} This chapter discusses the design and architecture of the system developed for competency extraction. It explains the implementation details and provides justification for the chosen approaches.
    
    \item \textbf{Evaluation:} This chapter presents the evaluation of the developed system. It compares baseline methods, evaluates the extraction methods and reinforcement learning enhancements, and measures efficiency and overall system performance.
    
    \item \textbf{Conclusion and Future Work:} The final chapter summarizes the contributions of the thesis and provides recommendations for future research directions.
\end{itemize}
"""
    elif 'Reinforcement Learning from Human Feedback (RLHF)' in subsection:
        return R"""RLHF: Let’s take it step by step
Reinforcement learning from Human Feedback (also referenced as RL from human preferences) is a challenging concept because it involves a multiple-model training process and different stages of deployment. In this blog post, we’ll break down the training process into three core steps:

Pretraining a language model (LM),
gathering data and training a reward model, and
fine-tuning the LM with reinforcement learning.
To start, we'll look at how language models are pretrained.

Pretraining language models
As a starting point RLHF use a language model that has already been pretrained with the classical pretraining objectives (see this blog post for more details). OpenAI used a smaller version of GPT-3 for its first popular RLHF model, InstructGPT. In their shared papers, Anthropic used transformer models from 10 million to 52 billion parameters trained for this task. DeepMind has documented using up to their 280 billion parameter model Gopher. It is likely that all these companies use much larger models in their RLHF-powered products.

This initial model can also be fine-tuned on additional text or conditions, but does not necessarily need to be. For example, OpenAI fine-tuned on human-generated text that was “preferable” and Anthropic generated their initial LM for RLHF by distilling an original LM on context clues for their “helpful, honest, and harmless” criteria. These are both sources of what we refer to as expensive, augmented data, but it is not a required technique to understand RLHF. Core to starting the RLHF process is having a model that responds well to diverse instructions.

In general, there is not a clear answer on “which model” is the best for the starting point of RLHF. This will be a common theme in this blog – the design space of options in RLHF training are not thoroughly explored.

Next, with a language model, one needs to generate data to train a reward model, which is how human preferences are integrated into the system.


Reward model training
Generating a reward model (RM, also referred to as a preference model) calibrated with human preferences is where the relatively new research in RLHF begins. The underlying goal is to get a model or system that takes in a sequence of text, and returns a scalar reward which should numerically represent the human preference. The system can be an end-to-end LM, or a modular system outputting a reward (e.g. a model ranks outputs, and the ranking is converted to reward). The output being a scalar reward is crucial for existing RL algorithms being integrated seamlessly later in the RLHF process.

These LMs for reward modeling can be both another fine-tuned LM or a LM trained from scratch on the preference data. For example, Anthropic has used a specialized method of fine-tuning to initialize these models after pretraining (preference model pretraining, PMP) because they found it to be more sample efficient than fine-tuning, but no one base model is considered the clear best choice for reward models.

The training dataset of prompt-generation pairs for the RM is generated by sampling a set of prompts from a predefined dataset (Anthropic’s data generated primarily with a chat tool on Amazon Mechanical Turk is available on the Hub, and OpenAI used prompts submitted by users to the GPT API). The prompts are passed through the initial language model to generate new text.

Human annotators are used to rank the generated text outputs from the LM. One may initially think that humans should apply a scalar score directly to each piece of text in order to generate a reward model, but this is difficult to do in practice. The differing values of humans cause these scores to be uncalibrated and noisy. Instead, rankings are used to compare the outputs of multiple models and create a much better regularized dataset.

There are multiple methods for ranking the text. One method that has been successful is to have users compare generated text from two language models conditioned on the same prompt. By comparing model outputs in head-to-head matchups, an Elo system can be used to generate a ranking of the models and outputs relative to each-other. These different methods of ranking are normalized into a scalar reward signal for training.

An interesting artifact of this process is that the successful RLHF systems to date have used reward language models with varying sizes relative to the text generation (e.g. OpenAI 175B LM, 6B reward model, Anthropic used LM and reward models from 10B to 52B, DeepMind uses 70B Chinchilla models for both LM and reward). An intuition would be that these preference models need to have similar capacity to understand the text given to them as a model would need in order to generate said text.

At this point in the RLHF system, we have an initial language model that can be used to generate text and a preference model that takes in any text and assigns it a score of how well humans perceive it. Next, we use reinforcement learning (RL) to optimize the original language model with respect to the reward model.


Fine-tuning with RL
Training a language model with reinforcement learning was, for a long time, something that people would have thought as impossible both for engineering and algorithmic reasons. What multiple organizations seem to have gotten to work is fine-tuning some or all of the parameters of a copy of the initial LM with a policy-gradient RL algorithm, Proximal Policy Optimization (PPO). Some parameters of the LM are frozen because fine-tuning an entire 10B or 100B+ parameter model is prohibitively expensive (for more, see Low-Rank Adaptation (LoRA) for LMs or the Sparrow LM from DeepMind) -- depending on the scale of the model and infrastructure being used. The exact dynamics of how many parameters to freeze, or not, is considered an open research problem. PPO has been around for a relatively long time – there are tons of guides on how it works. The relative maturity of this method made it a favorable choice for scaling up to the new application of distributed training for RLHF. It turns out that many of the core RL advancements to do RLHF have been figuring out how to update such a large model with a familiar algorithm (more on that later).

Let's first formulate this fine-tuning task as a RL problem. First, the policy is a language model that takes in a prompt and returns a sequence of text (or just probability distributions over text). The action space of this policy is all the tokens corresponding to the vocabulary of the language model (often on the order of 50k tokens) and the observation space is the distribution of possible input token sequences, which is also quite large given previous uses of RL (the dimension is approximately the size of vocabulary ^ length of the input token sequence). The reward function is a combination of the preference model and a constraint on policy shift.

The reward function is where the system combines all of the models we have discussed into one RLHF process. Given a prompt, x, from the dataset, the text y is generated by the current iteration of the fine-tuned policy. Concatenated with the original prompt, that text is passed to the preference model, which returns a scalar notion of “preferability”, 
r
θ
r 
θ
​
 . In addition, per-token probability distributions from the RL policy are compared to the ones from the initial model to compute a penalty on the difference between them. In multiple papers from OpenAI, Anthropic, and DeepMind, this penalty has been designed as a scaled version of the Kullback–Leibler (KL) divergence between these sequences of distributions over tokens, 
r
KL
r 
KL
​
 . The KL divergence term penalizes the RL policy from moving substantially away from the initial pretrained model with each training batch, which can be useful to make sure the model outputs reasonably coherent text snippets. Without this penalty the optimization can start to generate text that is gibberish but fools the reward model to give a high reward. In practice, the KL divergence is approximated via sampling from both distributions (explained by John Schulman here). The final reward sent to the RL update rule is 
r
=
r
θ
−
λ
r
KL
r=r 
θ
​
 −λr 
KL
​
 .

Some RLHF systems have added additional terms to the reward function. For example, OpenAI experimented successfully on InstructGPT by mixing in additional pre-training gradients (from the human annotation set) into the update rule for PPO. It is likely as RLHF is further investigated, the formulation of this reward function will continue to evolve.

Finally, the update rule is the parameter update from PPO that maximizes the reward metrics in the current batch of data (PPO is on-policy, which means the parameters are only updated with the current batch of prompt-generation pairs). PPO is a trust region optimization algorithm that uses constraints on the gradient to ensure the update step does not destabilize the learning process. DeepMind used a similar reward setup for Gopher but used synchronous advantage actor-critic (A2C) to optimize the gradients, which is notably different but has not been reproduced externally.


Technical detail note: The above diagram makes it look like both models generate different responses for the same prompt, but what really happens is that the RL policy generates text, and that text is fed into the initial model to produce its relative probabilities for the KL penalty. This initial model is untouched by gradient updates during training.

Optionally, RLHF can continue from this point by iteratively updating the reward model and the policy together. As the RL policy updates, users can continue ranking these outputs versus the model's earlier versions. Most papers have yet to discuss implementing this operation, as the deployment mode needed to collect this type of data only works for dialogue agents with access to an engaged user base. Anthropic discusses this option as Iterated Online RLHF (see the original paper), where iterations of the policy are included in the ELO ranking system across models. This introduces complex dynamics of the policy and reward model evolving, which represents a complex and open research question."""
    elif 'Direct Preference Optimization (DPO)' in subsection:
        return R"""Motivation
Fine tuning LLMs through instruction dataset and human-written completions can significantly enhance their performance in various tasks and ensure alignment with user intent. While instruction fine-tuning has shown promise, it often demands multiple experts to create completions. Another effective method involves leveraging human judgements to guide model refinement, where users determine their preferred completions, forming the basis for fine-tuning via approaches such as Reinforcement Learning with Human Feedback (RLHF), which notably requires only relative human judgment, making data collection more manageable.



However, this process entails two additional steps to fine-tune a pre-trained LLMs. Reinforcement learning, though powerful, demands substantial data input. Firstly, a reward model must be trained, followed by optimizing the pre-trained model alongside the reward model without drifting too far from the original pretrained model. Notably, such training may pose challenges on consumer-grade GPUs due to resource constraints, and it is a complex and often unstable procedure.



In contrast, DPO directly optimizes the model with a simple classification objective without the need of training a reward model.

How does DPO work?

Similar to RLHF, DPO requires a pre-trained LLM. However, the training process of DPO is conducted in a single step. We proceed directly to the loss function, which represents the core of DPO.

 L_DPO(pi_theta, pi_ref) = -E_{x, y_w, y_l}~D (log sigma(beta * log (pi_theta(y_w | x) / pi_ref(y_w | x)) - beta * log (pi_theta(y_l | x) / pi_ref(y_l | x)))
 
 
The dataset 
 comprises triplets, where 
 represents the input prompt, and 
 and 
 denote the 2 completions. 
 indicates the user's preferred completion, while 
 the non-preferred completion. 
 represents a non-negative constant, while 
 denotes the sigmoid activation function. And 
 is the expectation.

In the process of fine-tuning a pre-trained LLM, the approach may be analogous to RLHF. This entails the creation of an additional copy of the pre-trained LLM, one of which has its weights frozen, 
 , also known as the reference model, while the other, 
, is updated.

By defining the reward function as 

r_theta = \beta * log (pi_theta (y | x) / pi_ref (y | x))

 
we can see that, to minimize the loss function, the reward function has to assign a higher reward score to the preferred completion than to the non-preferred completion. By defining the reward this way, your language model is secretly a reward model.

This approach ensures that preferred completions are rewarded more favorably, thereby aligning with the main objective of the implicit reward model, which is to elevate the reward for preferred completions overall."""
    elif 'Parameter-Effizientes Fine-Tuning' in subsection:
        return R"""Challenges of fine tuning:
Fine-tuning LLM’s offers numerous benefits, but it also comes with significant challenges. Depending on the size of the model and the fine-tuning dataset, the process can take a significant amount of time and also high-performance GPUs or TPUs are often required to handle the computation load. LLM’s are large in size and storing the parameters of these models, especially when multiple versions are maintained (pre-trained and fine-tuned models), requires considerable storage capacity. When an LLM is fine-tuned on a specific task or dataset, the model can perform better in that area, losing its ability to perform well on more general tasks it was originally trained on.

Parameter-efficient Fine-tuning (PEFT):
Parameter-efficient Fine-tuning overcomes the problems of consumer hardware, storage costs by fine tuning only a small subset of model’s parameters significantly reducing the computational expenses while freezing the weights of original pretrained LLM. Additionally, this resolves the problem of catastrophic forgetting, which is a behavior seen when LLMs are fully adjusted.

When employing PEFT methods, the amount of storage required is only a few MBs for each downstream dataset, while still attaining performance comparable to full fine-tuning. For example, with full fine-tuning, 40GB of storage is required for each downstream dataset. The pretrained LLM is combined with the small trained weights from PEFT techniques and this model is used for numerous tasks. This method of fine tuning helps to get performance similar to full fine-tuning with less trainable parameters. There are several Parameter-efficient fine-tuning techniques and as follows:

Adapter
LoRA
Prefix tuning
Prompt tuning
P-tuning
IA3

In the below figure we can see that adapter layers are added after multi-head attention and feed-forward layers in the transformer architecture. The parameters of these added layers are only updated during fine-tuning while keeping the rest of the parameters frozen.

LoRA:
LoRA (Low-Rank Adaptation of Large Language Models) is a fine-tuning technique to train LLM’s on specific tasks or domains. This technique introduces trainable rank decomposition matrices into each layer of transformer architecture and also reduces trainable parameters for downstream task while keeping the pre trained weights frozen. LoRA papers says that this method can minimize the number of trainable parameters by up to 10,000 times and the GPU memory necessity by 3 times while still performing on par or better than fine-tuning model quality on various tasks.

Adaptors fine tuning method has Inference latency problem which is resolved by LoRA. It adds values to transformers instead of adding layers. A large matrix is expressed as the product of two smaller matrix in low-rank decomposition. This assumes that redundant information is often easily stored in a big matrix, especially in high-dimensional spaces.


Rather than altering the weight matrix W of a layer in all of its components, LoRA creates two smaller matrices, A and B, whose product roughly represents the modifications to W. The adaptation can be expressed mathematically as Y = W+AB, where A and B are the low-rank matrices. If W is an mxn matrix A might be mxr and B is rxn where r is rank and much smaller than m,n. During fine tuning only A and B are adjusted enabling the model to learn task specific features.

Read more about LoRA at Low-Rank Adaptation of Large Language Models

QLoRA:
QLoRA is the extended version of LoRA which works by quantizing the precision of the weight parameters in the pre trained LLM to 4-bit precision. Typically, parameters of trained models are stored in a 32-bit format, but QLoRA compresses them to a 4-bit format. This reduces the memory footprint of the LLM, making it possible to finetune it on a single GPU. This method significantly reduces the memory footprint, making it feasible to run LLM models on less powerful hardware, including consumer GPUs.


According to QLoRA paper:

QLORA introduces multiple innovations designed to reduce memory use without sacrificing performance: (1) 4-bit NormalFloat, an information theoretically optimal quantization data type for normally distributed data that yields better empirical results than 4-bit Integers and 4-bit Floats. (2) Double Quantization, a method that quantizes the quantization constants, saving an average of about 0.37 bits per parameter (approximately 3 GB for a 65B model). (3) Paged Optimizers, using NVIDIA unified memory to avoid the gradient checkpointing memory spikes that occur when processing a mini-batch with a long sequence length.

The above figure shows the result of LLaMA 2 7B model trained on different floating points and results of models on various tasks. The model trained on NF4 and float 4-bit gives better results than LoRA and LLaMA 2 7B base model, while 4-bit NormalFloat perform slightly better performance than float4 datatype. QLoRA decreases the memory requirements by almost using NF4 type. However, the tradeoff is a slower training time, which is to be expected due to the quantization and dequantization steps.
The above figure shows the result of LLaMA 2 7B model trained on different floating points and results of models on various tasks. The model trained on NF4 and float 4-bit gives better results than LoRA and LLaMA 2 7B base model, while 4-bit NormalFloat perform slightly better performance than float4 datatype. QLoRA decreases the memory requirements by almost using NF4 type. However, the tradeoff is a slower training time, which is to be expected due to the quantization and dequantization steps.

4-bit Normal Float:
NF4 is a data type specifically designed for AI applications, particularly in the context of quantizing the weights of neural networks to reduce memory footprints of models significantly while attempting to maintain performance. This is crucial for deploying large models on less powerful hardware​. NF4 is information-theoretically optimal for data that has a normal distribution, which is a common feature of neural network weights and can represent these weights more accurately than a standard 4-bit float would, within the given bit constraint.

While standard 4-bit Float is more general-purpose floating-point representation and not specifically optimized for specific applications. 4-bit float has very limited precision and range and due to its limitations in precision and range, standard 4-bit floats are less common in AI and machine learning applications, especially for tasks requiring high precision in calculations.

Let’s see how the given number is stored in floating point for various floating points datatypes.


Each floating point contains 3 different parts which store details about the stored number, that is sign, exponent and fraction also known as mantissa for the given number. The number is first converted into binary format and then stored in the datatype. Each datatype differs in the number of bits they use and hence in their precision and range. For example, FP32 can represent numbers approximately between ±1.18×10^-38 and ±3.4×10³⁸. FP32 is single-precision binary floating point format and has been used as the default format to store weights and biases in Deep Learning. while Fp8 has a range of [-127, 127] and NF4 has a range of [-8, 7].

QLoRA uses brainfloat16 (bfloat16) datatype to perform computational operation that is during forward and backward pass. Brain Floating Point was developed by Google for use in machine learning and other applications that require high throughput of floating-point operations.

Quantization:
Quantization is a technique that is helpful in reducing the size of the model by converting high precision data to low precision. In simple terms, it converts datatype of high bits to fewer bits. For example, converting FP32 to 8-bit Integers is a quantization technique."""


if __name__ == '__main__':
    total_subsections = sum(len(chunked_file(file)) for file in SECTION_FILES)
    print(f'Total number of subsections: {total_subsections}')
    print('Evaluating all the subsections will take approximately', total_subsections * 2, 'minutes.')

    for name, file in zip(SECTIONS, SECTION_FILES):
        open_browser()

        section_overview = load_file(file)

        for subsection in tqdm(chunked_file(file), desc=f'Processing {name}'):
            description = load_file(f'{FOLDER}\\{DATA}')
            additional_information = get_additional_information(name, subsection)
            additional_information_text = (
                f'\n\nZusätzliche Informationen die für die Bearbeitung des Abschnitts hilfreich sein könnten:\n{additional_information}'
                if additional_information
                else ''
            )

            # Wenn die komplett Projekt beschreibung noch drinnen sein soll, dann uncomment this query = f"""Beschreibung des Projekts: {description}
            query_write = f"""{additional_information_text}


Struktur der Arbeit:  
{STRUKTUR}

Wir bearbeiten nun das Kapitel {name}:  
{section_overview}

---

Deine Aufgabe:  
Bearbeite ausschließlich den folgenden Abschnitt basierend auf den bereitgestellten Stichpunkten und Informationen:

Unterabschnitt:  
{subsection}

---

Inhaltliche Anforderungen:

1. Zielgruppe: Die Zielgruppe sind Personen, die über grundlegende Kenntnisse verfügen, aber nicht tief im Thema stecken (z. B. Betreuende).
2. Umfassende Abdeckung: Stelle sicher, dass alle in den Notizen angegebenen Informationen in der Ausarbeitung vorkommen. Sollten sich Informationen wiederholen, müssen diese nicht mehrfach erwähnt werden, aber es darf nichts aus den Notizen fehlen.
3. Kritische Reflexion: Gehe kritisch auf die Ergebnisse und deren Bedeutung ein.
4. TODO-Einträge: Falls eine Grafik fehlt oder in den Notizen noch ein TODO steht, dann füge an dieser Stelle ein `\\todo{{}}` ein. Beispiel: `\\todo{{Grafik zur Veranschaulichung der Systemarchitektur einfügen}}`.

---

Formale Anforderungen:

1. LaTeX-Formatierung: Alle Antworten werden in deutscher Sprache und LaTeX formatiert, um den akademischen Standards zu entsprechen.
2. Konsistente Terminologie: Verwende eine einheitliche Terminologie für dasselbe Konzept. Beispielsweise sollte das Wort „System“ immer verwendet werden, anstatt es durch Begriffe wie „Computer“ oder „Server“ auszutauschen.
3. Klarheit und Präzision: Vermeide Verstärker wie „sehr“ und ähnliche Wörter. Begründe, relativiere oder vermeide die Verwendung von Superlativen und absoluten/faktischen Aussagen.
4. Vermeidung von Füllwörtern: Verzichte auf Füllwörter, die den Inhalt nicht voranbringen, wie z. B. „oft“, „häufig“, „regelmäßig“, „typischerweise“, „gerade“, „nur“, „so“, „daher“, „schließlich“ usw.
5. Schreibstil: Schreibe vorzugsweise in der dritten Person oder im Passiv, um die akademische Formalität zu wahren. Verwende z. B. „die Arbeit“ anstelle von „unsere Arbeit“.
6. Konsistenz im Stil: Halte den Schreibstil im gesamten Abschnitt konsistent."""

            query_check = f"""Prüfe die folgende wissenschaftliche Ausarbeitung für die Subsection meiner Thesis. Achte darauf, dass ALLE, wirklich ALLE Informationen aus den Notizen in die Ausarbeitung integriert sind. Wenn Informationen fehlen, liste diese vollständig auf und begründe, warum sie fehlen. Wenn alles enthalten ist, bestätige dies mit "Ja, alles enthalten". Falls etwas fehlt, antworte mit "Nein, folgende Informationen fehlen: [Liste an fehlenden Informationen hier]".

Hier sind die Informationen zur Subsection:
---
{subsection}
---
Stelle sicher, dass alle Punkte, die in den Notizen definiert sind, mindestens einmal vorkommen und alle relevanten Aspekte wissenschaftlich präzise behandelt werden. Verwende präzise Begründungen für eventuelle Lücken in der Ausarbeitung."""
            new_tab()
            query_chat(query_write)
            query_chat(query_check)


# TODO future - Führen Sie Abkürzungen einmal ein und verwenden Sie sie konsequent mit den Befehlen \gls{} und \glspl{}.
# TODO future - Fügen Sie die Quellenangaben in die Fußzeile ein, um die Arbeit wissenschaftlich korrekt zu zitieren.
# TODO future - Adherence to the Wissenschaftliches_Arbeiten.pdf guidelines
"""
Output-Spezifikationen:
Alle Antworten werden in Deutesch verfasst und in LaTeX formatiert, um den akademischen Standards zu entsprechen und den Inhalt korrekt wiederzugeben.

- Vermeiden Sie die Verwendung von Verstärkern wie „sehr“ und ähnlichen Wörtern.
- Begründen, relativieren oder vermeiden Sie die Verwendung von Superlativen und absoluten/faktischen Aussagen.
- Vermeiden Sie Füllwörter, die den Inhalt der Sätze nicht beeinflussen, wie z. B. „oft“, „häufig“, „regelmäßig“, „typischerweise“, „gerade“, „nur“, „so“, „daher“, „schließlich“ usw.
- Verwenden Sie eine einheitliche Terminologie für ein und dasselbe Konzept, auch wenn dies eintönig erscheinen mag; sprechen Sie immer von „dem System“ und nicht austauschbar mit „Computer“, „Server“, usw.
- Schreiben Sie vorzugsweise in der dritten Person oder im Passiv, um die akademische Formalität zu wahren (z. B. „die Arbeit“ statt „unsere Arbeit“).

"""


def get_assistant_response(conv: dict) -> str:
    messages: list[tuple[float, str]] = [
        (message['message']['create_time'], message['message']['content']['parts'][0])
        for message in conv['mapping'].values()
        if message['message'] and message['message']['author']['role'] == 'assistant'
    ]

    assert messages, f'No assistant response found in conversation: {conv}'

    # return a string with newlines between each message, sorted by oldest to newest
    return '\n\n\n\n\n\n'.join(part for time, part in sorted(messages, key=lambda x: x[0]))


def extract_response_if_matches(section_overview: str, subsection: str, conv) -> str | None:
    for message in conv['mapping'].values():
        if message['message'] is None:
            continue
        message_content = '\n\n'.join(part for part in message['message']['content']['parts'] if isinstance(part, str))
        if section_overview in message_content:
            filtered_message_content = message_content.replace(section_overview, '')
            if subsection in filtered_message_content:
                return get_assistant_response(conv)


if __name__ == '__main__2':
    CONVERSATION_PATH = R'C:\Users\berti\Downloads\conversations.json'
    conversations = json.loads(load_file(CONVERSATION_PATH))
    # with open(CONVERSATION_PATH, 'w') as f:
    #     json.dump(conversations, f, indent=4, ensure_ascii=False)

    # filter out conversations which are not from today by the create_time timestamp
    today_timestamp = time.time() - 24 * 60 * 60
    conversations = [conv for conv in conversations if conv['create_time'] > today_timestamp]

    for name, file, section_index in zip(SECTIONS, SECTION_FILES, range(1, len(SECTIONS) + 1)):
        section_overview = load_file(file)

        for subsection_index, subsection in enumerate(tqdm(chunked_file(file), desc=f'Processing {name}'), start=1):
            for conv in conversations:
                if response := extract_response_if_matches(section_overview, subsection, conv):
                    with open(f'{FOLDER}\\Stichpunkte\\{section_index:02d}.{subsection_index:02d}.txt', 'w') as f:
                        f.write(response)
                    break
            else:
                print('No response found for')
