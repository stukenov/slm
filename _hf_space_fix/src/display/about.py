from src.display.utils import ModelType

TITLE = """<h1 style="text-align:left;float:left; id="space-title">🤗 Kaz LLM Leaderboard</h1> <h3 style="text-align:left;float:left;> Track, rank and evaluate open LLMs and chatbots </h3>"""

INTRODUCTION_TEXT = """
"""

icons = f"""
- {ModelType.PT.to_str(" : ")} model: new, base models, trained on a given text corpora using masked modelling
- {ModelType.CPT.to_str(" : ")} model: new, base models, continuously trained on further corpus (which may include IFT/chat data) using masked modelling
- {ModelType.FT.to_str(" : ")} model: pretrained models finetuned on more data
- {ModelType.chat.to_str(" : ")} model: chat like fine-tunes, either using IFT (datasets of task instruction), RLHF or DPO (changing the model loss a bit with an added policy), etc
- {ModelType.merges.to_str(" : ")} model: merges or MoErges, models which have been merged or fused without additional fine-tuning.
"""

LLM_BENCHMARKS_TEXT = """
## En:

Kaz LLM is a benchmark for LLM with multiple-choice tasks on the following topics:
- mmlu-translated-kk
- kazakh-constitution-mc
- kazakh-dastur-mc
- kazakh-unified-national-testing-mc(biology,english, geography, history of kz, world's history, society rights, literature, kazakh language)

Each task contains from 4 to 8 answer choices.
## Instructions for Use
### Installation
To install the necessary library, run the following command:
```bash
git clone https://github.com/horde-research/horde-common.git
cd scripts
pip install -r requirements.txt
```
### Execution
To run the benchmark, use the following command:
```bash
python mc-eval-simplified-inference.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --output_path .
```
### Results
After executing the above command, a JSON file will be created in the `output` directory, which must be attached. This file contains the results of the tasks and a description of the session, and **must not be modified**.
## Anti-Cheating Policy
If cheating or attempts to modify the output file are detected, we reserve the right to delete your submission.
Thank you for participating!


## Қаз:

Kaz LLM – бұл төмендегі тақырыптар бойынша көп нұсқалы тапсырмалармен жұмыс істейтін LLM үшін арналған бенчмарк:
- mmlu-translated-kk
- kazakh-constitution-mc
- kazakh-dastur-mc
- kazakh-unified-national-testing-mc(biology,english, geography, history of kz, world's history, society rights, literature, kazakh language)

Әр тапсырмада 4-8 жауап нұсқасы бар.

## Қолдану бойынша нұсқаулық
### Орнату
Қажетті кітапхананы орнату үшін төмендегі команданы орындаңыз:
```bash
git clone https://github.com/horde-research/horde-common.git
cd scripts
pip install -r requirements.txt
```

### Орындау
Бенчмаркті іске қосу үшін келесі команданы пайдаланыңыз:
```bash
python mc-eval-simplified-inference.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --output_path .
```

### Нәтижелер
Жоғарыдағы команданы орындағаннан кейін `output` директориясында JSON файлы жасалады. Бұл файл тапсырмалар нәтижелерін және сессия сипаттамасын қамтиды, және **өзгертілмеуі тиіс**.

## Анти-алаяқтық саясаты
Егер файлды өзгертулер немесе алаяқтық әрекеттер анықталса, біз сіздің жұмысыңызды жою құқығын өзімізде қалдырамыз.  
Қатысқаныңыз үшін рақмет!

"""

FAQ_TEXT = """

## SUBMISSIONS
My model requires `trust_remote_code=True`, can I submit it?
- *We only support models that have been integrated into a stable version of the `transformers` library for automatic submission, as we don't want to run possibly unsafe code on our cluster.*

What about models of type X?
- *We only support models that have been integrated into a stable version of the `transformers` library for automatic submission.*

How can I follow when my model is launched?
- *You can look for its request file [here](https://huggingface.co/datasets/open-llm-leaderboard/requests) and follow the status evolution, or directly in the queues above the submit form.*

My model disappeared from all the queues, what happened?
- *A model disappearing from all the queues usually means that there has been a failure. You can check if that is the case by looking for your model [here](https://huggingface.co/datasets/open-llm-leaderboard/requests).*

What causes an evaluation failure?
- *Most of the failures we get come from problems in the submissions (corrupted files, config problems, wrong parameters selected for eval ...), so we'll be grateful if you first make sure you have followed the steps in `About`. However, from time to time, we have failures on our side (hardware/node failures, problems with an update of our backend, connectivity problems ending up in the results not being saved, ...).*

How can I report an evaluation failure?
- *As we store the logs for all models, feel free to create an issue, **where you link to the requests file of your model** (look for it [here](https://huggingface.co/datasets/open-llm-leaderboard/requests/tree/main)), so we can investigate! If the model failed due to a problem on our side, we'll relaunch it right away!*
*Note: Please do not re-upload your model under a different name, it will not help*

---------------------------

## RESULTS
What kind of information can I find?
- *Let's imagine you are interested in the Yi-34B results. You have access to 3 different information categories:*
      - *The [request file](https://huggingface.co/datasets/open-llm-leaderboard/requests/blob/main/01-ai/Yi-34B_eval_request_False_bfloat16_Original.json): it gives you information about the status of the evaluation*
      - *The [aggregated results folder](https://huggingface.co/datasets/open-llm-leaderboard/results/tree/main/01-ai/Yi-34B): it gives you aggregated scores, per experimental run*
      - *The [details dataset](https://huggingface.co/datasets/open-llm-leaderboard/details_01-ai__Yi-34B/tree/main): it gives you the full details (scores and examples for each task and a given model)*


Why do models appear several times in the leaderboard?
- *We run evaluations with user-selected precision and model commit. Sometimes, users submit specific models at different commits and at different precisions (for example, in float16 and 4bit to see how quantization affects performance). You should be able to verify this by displaying the `precision` and `model sha` columns in the display. If, however, you see models appearing several times with the same precision and hash commit, this is not normal.*

What is this concept of "flagging"?
- *This mechanism allows users to report models that have unfair performance on the leaderboard. This contains several categories: exceedingly good results on the leaderboard because the model was (maybe accidentally) trained on the evaluation data, models that are copies of other models not attributed properly, etc.*

My model has been flagged improperly, what can I do?
- *Every flagged model has a discussion associated with it - feel free to plead your case there, and we'll see what to do together with the community.*

---------------------------

## HOW TO SEARCH FOR A MODEL
Search for models in the leaderboard by:
1. Name, e.g., *model_name*
2. Multiple names, separated by `;`, e.g., *model_name1;model_name2*
3. License, prefix with `license:`, e.g., *license: MIT*
4. Combination of name and license, order is irrelevant, e.g., *model_name; license: cc-by-sa-4.0*

---------------------------

## EDITING SUBMISSIONS
I upgraded my model and want to re-submit, how can I do that?
- *Please open an issue with the precise name of your model, and we'll remove your model from the leaderboard so you can resubmit. You can also resubmit directly with the new commit hash!*

I need to rename my model, how can I do that?
- *You can use @Weyaxi 's [super cool tool](https://huggingface.co/spaces/Weyaxi/open-llm-leaderboard-renamer) to request model name changes, then open a discussion where you link to the created pull request, and we'll check them and merge them as needed.*

---------------------------

## OTHER
Why do you differentiate between pretrained, continuously pretrained, fine-tuned, merges, etc?
- *These different models do not play in the same categories, and therefore need to be separated for fair comparison. Base pretrained models are the most interesting for the community, as they are usually good models to fine-tune later on - any jump in performance from a pretrained model represents a true improvement on the SOTA.
Fine-tuned and IFT/RLHF/chat models usually have better performance, but the latter might be more sensitive to system prompts, which we do not cover at the moment in the Open LLM Leaderboard.
Merges and moerges have artificially inflated performance on test sets, which is not always explainable, and does not always apply to real-world situations.*

What should I use the leaderboard for?
- *We recommend using the leaderboard for 3 use cases: 1) getting an idea of the state of open pretrained models, by looking only at the ranks and score of this category; 2) experimenting with different fine-tuning methods, datasets, quantization techniques, etc, and comparing their score in a reproducible setup, and 3) checking the performance of a model of interest to you, wrt to other models of its category.*

Why don't you display closed-source model scores?
- *This is a leaderboard for Open models, both for philosophical reasons (openness is cool) and for practical reasons: we want to ensure that the results we display are accurate and reproducible, but 1) commercial closed models can change their API thus rendering any scoring at a given time incorrect 2) we re-run everything on our cluster to ensure all models are run on the same setup and you can't do that for these models.*

I have an issue with accessing the leaderboard through the Gradio API
- *Since this is not the recommended way to access the leaderboard, we won't provide support for this, but you can look at tools provided by the community for inspiration!*

I have another problem, help!
- *Please open an issue in the discussion tab, and we'll do our best to help you in a timely manner :) *
"""


EVALUATION_QUEUE_TEXT = f"""
# Evaluation Queue for the 🤗 Open LLM Leaderboard

Models added here will be automatically evaluated on the 🤗 cluster.

## Don't forget to read the FAQ and the About tabs for more information!

## First steps before submitting a model

### 1) Make sure you can load your model and tokenizer using AutoClasses:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
If this step fails, follow the error messages to debug your model before submitting it. It's likely your model has been improperly uploaded.

Note: make sure your model is public!
Note: if your model needs `use_remote_code=True`, we do not support this option yet but we are working on adding it, stay posted!

### 2) Convert your model weights to [safetensors](https://huggingface.co/docs/safetensors/index)
It's a new format for storing weights which is safer and faster to load and use. It will also allow us to add the number of parameters of your model to the `Extended Viewer`!

### 3) Make sure your model has an open license!
This is a leaderboard for Open LLMs, and we'd love for as many people as possible to know they can use your model 🤗

### 4) Fill up your model card
When we add extra information about models to the leaderboard, it will be automatically taken from the model card

### 5) Select the correct precision
Not all models are converted properly from `float16` to `bfloat16`, and selecting the wrong precision can sometimes cause evaluation error (as loading a `bf16` model in `fp16` can sometimes generate NaNs, depending on the weight range).

<b>Note:</b> Please be advised that when submitting, git <b>branches</b> and <b>tags</b> will be strictly tied to the <b>specific commit</b> present at the time of submission. This ensures revision consistency.
## Model types
{icons}
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
@misc{open-llm-leaderboard,
  author = {Edward Beeching and Clémentine Fourrier and Nathan Habib and Sheon Han and Nathan Lambert and Nazneen Rajani and Omar Sanseviero and Lewis Tunstall and Thomas Wolf},
  title = {Open LLM Leaderboard},
  year = {2023},
  publisher = {Hugging Face},
  howpublished = "\url{https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard}"
}
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
@misc{clark2018think,
      title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
      author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
      year={2018},
      eprint={1803.05457},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
@misc{zellers2019hellaswag,
      title={HellaSwag: Can a Machine Really Finish Your Sentence?},
      author={Rowan Zellers and Ari Holtzman and Yonatan Bisk and Ali Farhadi and Yejin Choi},
      year={2019},
      eprint={1905.07830},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{hendrycks2021measuring,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      year={2021},
      eprint={2009.03300},
      archivePrefix={arXiv},
      primaryClass={cs.CY}
}
@misc{lin2022truthfulqa,
      title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
      author={Stephanie Lin and Jacob Hilton and Owain Evans},
      year={2022},
      eprint={2109.07958},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{DBLP:journals/corr/abs-1907-10641,
      title={{WINOGRANDE:} An Adversarial Winograd Schema Challenge at Scale},
      author={Keisuke Sakaguchi and Ronan Le Bras and Chandra Bhagavatula and Yejin Choi},
      year={2019},
      eprint={1907.10641},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{DBLP:journals/corr/abs-2110-14168,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and
                  Vineet Kosaraju and
                  Mohammad Bavarian and
                  Mark Chen and
                  Heewoo Jun and
                  Lukasz Kaiser and
                  Matthias Plappert and
                  Jerry Tworek and
                  Jacob Hilton and
                  Reiichiro Nakano and
                  Christopher Hesse and
                  John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
