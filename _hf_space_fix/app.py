import logging
import os
os.makedirs("tmp", exist_ok=True)
os.environ['TMP_DIR'] = "tmp"
import glob
import gradio as gr
import json
from io import BytesIO

from src.radial.radial import create_plot
from src.display.css_html_js import custom_css
from src.envs import API, H4_TOKEN, HF_HOME, REPO_ID, RESET_JUDGEMENT_ENV
from src.leaderboard.build_leaderboard import build_leadearboard_df, download_openbench, download_dataset, invalidate_cache
import huggingface_hub


os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── Friendly benchmark names ───
BENCHMARK_LABELS = {
    "mmlu_translated_kk": "MMLU (KK)",
    "kk_constitution_mc": "Constitution",
    "kk_dastur_mc": "Dastur",
    "kazakh_and_literature_unt_mc": "Literature",
    "kk_geography_unt_mc": "Geography",
    "kk_world_history_unt_mc": "World History",
    "kk_history_of_kazakhstan_unt_mc": "KZ History",
    "kk_english_unt_mc": "English",
    "kk_biology_unt_mc": "Biology",
    "kk_human_society_rights_unt_mc": "Society & Rights",
}

BENCHMARK_COLS = list(BENCHMARK_LABELS.keys())


def score_class(val):
    """CSS class based on score value."""
    if val >= 0.7:
        return "score-high"
    if val >= 0.4:
        return "score-mid"
    return "score-low"


def build_leaderboard_html(df):
    """Render the leaderboard DataFrame as a clean HTML table."""
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        rank = i
        if rank <= 3:
            rank_cls = f"rank-{rank}"
        else:
            rank_cls = "rank-other"

        model_name = row["model"]
        dtype = row.get("model_dtype", "")
        if dtype and dtype != "0":
            dtype_str = str(dtype).replace("torch.", "")
            badge = f'<span class="model-type-badge">{dtype_str}</span>'
        else:
            badge = ""

        avg = row["avg"]
        avg_pct = f"{avg * 100:.1f}"

        score_cells = ""
        for col in BENCHMARK_COLS:
            v = row[col]
            pct = f"{v * 100:.1f}"
            cls = score_class(v)
            score_cells += f'<td class="num {cls}">{pct}</td>'

        rows.append(f"""<tr>
            <td><span class="rank-badge {rank_cls}">{rank}</span></td>
            <td><span class="model-name">{model_name}</span>{badge}</td>
            <td class="num avg-cell">{avg_pct}</td>
            {score_cells}
        </tr>""")

    header_cells = ""
    for col in BENCHMARK_COLS:
        label = BENCHMARK_LABELS[col]
        header_cells += f'<th class="num">{label}</th>'

    html = f"""<div id="leaderboard-table" style="overflow-x:auto;">
    <table>
        <thead>
            <tr>
                <th style="width:50px;">#</th>
                <th>Model</th>
                <th class="num">Avg %</th>
                {header_cells}
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    </div>"""
    return html


def handle_file_upload(file_path):
    logging.info("File uploaded: %s", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        v = json.load(f)
    return v

def submit_file(v, mn):
    print('START SUBMITTING!!!')

    if 'results' not in v:
        return "Invalid JSON: missing 'results' key"

    new_file = v['results']
    new_file['model'] = mn

    columns = [
        'mmlu_translated_kk', 'kk_constitution_mc', 'kk_dastur_mc',
        'kazakh_and_literature_unt_mc', 'kk_geography_unt_mc',
        'kk_world_history_unt_mc', 'kk_history_of_kazakhstan_unt_mc',
        'kk_english_unt_mc', 'kk_biology_unt_mc', 'kk_human_society_rights_unt_mc'
    ]

    for column in columns:
        if column not in new_file or not isinstance(new_file[column], dict):
            return f"Missing or invalid column: {column}"
        if 'acc,none' not in new_file[column]:
            return f"Missing 'acc,none' key in column: {column}"
        new_file[column] = new_file[column]['acc,none']

    if 'config' not in v or 'model_dtype' not in v['config']:
        return "Missing 'config' or 'model_dtype' in JSON"

    new_file['model_dtype'] = v['config']["model_dtype"]
    new_file['ppl'] = 0

    print('WE READ FILE: ', new_file)

    buf = BytesIO()
    buf.write(json.dumps(new_file).encode('utf-8'))
    buf.seek(0)
    API.upload_file(
        path_or_fileobj=buf,
        path_in_repo="model_data/external/" + mn.replace('/', '__') + ".json",
        repo_id="stukenov/s-openbench-eval",
        repo_type="dataset",
    )

    os.environ[RESET_JUDGEMENT_ENV] = "1"
    return "Success!"


def restart_space():
    API.restart_space(repo_id=REPO_ID)

def update_plot(selected_models):
    return create_plot(selected_models)


HEADER_HTML = """
<div id="header-block">
    <h1>Kaz LLM Leaderboard</h1>
    <p>Evaluating language models on Kazakh-language benchmarks</p>
</div>
"""

SUBMIT_INTRO = """
### How to submit

1. Clone the evaluation repo and run the benchmark:
```bash
git clone https://github.com/horde-research/horde-common.git
cd scripts && pip install -r requirements.txt
python mc-eval-simplified-inference.py --model_id your-model --output_path .
```

2. Upload the resulting JSON file below with your model name.

The output file **must not be modified**. Cheating attempts will result in removal.
"""


def build_demo():
    demo = gr.Blocks(title="Kaz LLM Leaderboard", css=custom_css)
    leaderboard_df = build_leadearboard_df()

    with demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs(elem_classes="tab-buttons"):
            # ── Tab 1: Leaderboard ──
            with gr.TabItem("Leaderboard", id=0):
                leaderboard_html = build_leaderboard_html(leaderboard_df)
                gr.HTML(leaderboard_html)

            # ── Tab 2: Submit ──
            with gr.TabItem("Submit", id=1):
                with gr.Column(elem_id="submit-section"):
                    gr.Markdown(SUBMIT_INTRO, elem_classes="instructions-block")

                    model_name_textbox = gr.Textbox(label="Model name", placeholder="e.g. deepseek-ai/DeepSeek-R1")
                    file_output = gr.File(label="Drop your JSON results here", type="filepath")
                    uploaded_file = gr.State()
                    out = gr.Textbox(label="Status", interactive=False, elem_id="submit-status")
                    submit_button = gr.Button("Submit", variant='primary', elem_id="submit-btn")

                    file_output.upload(
                        fn=handle_file_upload,
                        inputs=file_output,
                        outputs=uploaded_file
                    )
                    submit_button.click(
                        fn=submit_file,
                        inputs=[uploaded_file, model_name_textbox],
                        outputs=[out]
                    )

            # ── Tab 3: Analytics ──
            with gr.TabItem("Analytics", id=2):
                with gr.Column(elem_id="analytics-section"):
                    model_dropdown = gr.Dropdown(
                        choices=leaderboard_df["model"].tolist(),
                        label="Select models to compare",
                        value=leaderboard_df["model"].tolist(),
                        multiselect=True,
                    )
                    plot = gr.Plot(update_plot(model_dropdown.value))
                    model_dropdown.change(
                        fn=update_plot,
                        inputs=[model_dropdown],
                        outputs=[plot]
                    )

    return demo


def aggregate_leaderboard_data():
    download_dataset("stukenov/s-openbench-eval", "m_data")  # only external submissions

    data_list = [
        {
            "model_dtype": "torch.float16",
            "model": "dummy-random-baseline",
            "ppl": 0,
            "mmlu_translated_kk": 0.22991508817766165,
            "kk_constitution_mc": 0.25120772946859904,
            "kk_dastur_mc": 0.24477611940298508,
            "kazakh_and_literature_unt_mc": 0.2090443686006826,
            "kk_geography_unt_mc": 0.2019790454016298,
            "kk_world_history_unt_mc": 0.1986970684039088,
            "kk_history_of_kazakhstan_unt_mc": 0.19417177914110428,
            "kk_english_unt_mc": 0.189804278561675,
            "kk_biology_unt_mc": 0.22330729166666666,
            "kk_human_society_rights_unt_mc": 0.242152466367713,
        },
        {
            "model_dtype": "torch.float16",
            "model": "gpt-4o-mini",
            "ppl": 0,
            "mmlu_translated_kk": 0.5623775310254735,
            "kk_constitution_mc": 0.79,
            "kk_dastur_mc": 0.755,
            "kazakh_and_literature_unt_mc": 0.4953071672354949,
            "kk_geography_unt_mc": 0.5675203725261933,
            "kk_world_history_unt_mc": 0.6091205211726385,
            "kk_history_of_kazakhstan_unt_mc": 0.47883435582822087,
            "kk_english_unt_mc": 0.6763768775603095,
            "kk_biology_unt_mc": 0.607421875,
            "kk_human_society_rights_unt_mc": 0.7309417040358744,
        },
        {
            "model_dtype": "api",
            "model": "gpt-4o",
            "ppl": 0,
            "mmlu_translated_kk": 0.7419986936642717,
            "kk_constitution_mc": 0.841,
            "kk_dastur_mc": 0.798,
            "kazakh_and_literature_unt_mc": 0.6785409556313993,
            "kk_geography_unt_mc": 0.629802095459837,
            "kk_world_history_unt_mc": 0.6783387622149837,
            "kk_history_of_kazakhstan_unt_mc": 0.6785276073619632,
            "kk_english_unt_mc": 0.7410104688211198,
            "kk_biology_unt_mc": 0.6979166666666666,
            "kk_human_society_rights_unt_mc": 0.7937219730941704,
        },
        {
            "model_dtype": "torch.float16",
            "model": "nova-pro-v1",
            "ppl": 0,
            "mmlu_translated_kk": 0.6792945787067276,
            "kk_constitution_mc": 0.7753623188405797,
            "kk_dastur_mc": 0.718407960199005,
            "kazakh_and_literature_unt_mc": 0.4656569965870307,
            "kk_geography_unt_mc": 0.5541327124563445,
            "kk_world_history_unt_mc": 0.6425081433224755,
            "kk_history_of_kazakhstan_unt_mc": 0.5,
            "kk_english_unt_mc": 0.6845698680018206,
            "kk_biology_unt_mc": 0.6197916666666666,
            "kk_human_society_rights_unt_mc": 0.7713004484304933,
        },
        {
            "model_dtype": "torch.float16",
            "model": "gemini-1.5-pro",
            "ppl": 0,
            "mmlu_translated_kk": 0.7380796864794252,
            "kk_constitution_mc": 0.8164251207729468,
            "kk_dastur_mc": 0.7383084577114428,
            "kazakh_and_literature_unt_mc": 0.5565273037542662,
            "kk_geography_unt_mc": 0.6065192083818394,
            "kk_world_history_unt_mc": 0.6669381107491856,
            "kk_history_of_kazakhstan_unt_mc": 0.5791411042944785,
            "kk_english_unt_mc": 0.7114246700045517,
            "kk_biology_unt_mc": 0.6673177083333334,
            "kk_human_society_rights_unt_mc": 0.7623318385650224,
        },
        {
            "model_dtype": "torch.float16",
            "model": "gemini-1.5-flash",
            "ppl": 0,
            "mmlu_translated_kk": 0.6335728282168517,
            "kk_constitution_mc": 0.748792270531401,
            "kk_dastur_mc": 0.7054726368159204,
            "kazakh_and_literature_unt_mc": 0.4761092150170648,
            "kk_geography_unt_mc": 0.5640279394644936,
            "kk_world_history_unt_mc": 0.5838762214983714,
            "kk_history_of_kazakhstan_unt_mc": 0.43374233128834355,
            "kk_english_unt_mc": 0.6681838871187984,
            "kk_biology_unt_mc": 0.6217447916666666,
            "kk_human_society_rights_unt_mc": 0.7040358744394619,
        },
        {
            "model_dtype": "torch.float16",
            "model": "claude-3-5-sonnet",
            "ppl": 0,
            "mmlu_translated_kk": 0.7335075114304376,
            "kk_constitution_mc": 0.8623188405797102,
            "kk_dastur_mc": 0.7950248756218905,
            "kazakh_and_literature_unt_mc": 0.6548634812286689,
            "kk_geography_unt_mc": 0.6431897555296857,
            "kk_world_history_unt_mc": 0.6669381107491856,
            "kk_history_of_kazakhstan_unt_mc": 0.6251533742331289,
            "kk_english_unt_mc": 0.7291761492944925,
            "kk_biology_unt_mc": 0.6686197916666666,
            "kk_human_society_rights_unt_mc": 0.8026905829596412,
        },
        {
            "model_dtype": "torch.float16",
            "model": "yandex-gpt",
            "ppl": 0,
            "mmlu_translated_kk": 0.39777922926192033,
            "kk_constitution_mc": 0.7028985507246377,
            "kk_dastur_mc": 0.6159203980099502,
            "kazakh_and_literature_unt_mc": 0.3914249146757679,
            "kk_geography_unt_mc": 0.4912689173457509,
            "kk_world_history_unt_mc": 0.5244299674267101,
            "kk_history_of_kazakhstan_unt_mc": 0.4030674846625767,
            "kk_english_unt_mc": 0.5844333181611289,
            "kk_biology_unt_mc": 0.4368489583333333,
            "kk_human_society_rights_unt_mc": 0.6995515695067265,
        },
    ]

    files_list = glob.glob("./m_data/model_data/external/*.json")
    logging.info(f'FILES LIST: {files_list}')

    for file in files_list:
        logging.info(f'Trying to read external submit file: {file}')
        try:
            with open(file) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logging.warning(f"File {file} is not a dict, skipping")
                continue
            required_keys = {'model_dtype', 'model', 'ppl', 'mmlu_translated_kk'}
            if not required_keys.issubset(data.keys()):
                logging.warning(f"File {file} missing required keys, skipping")
                continue

            logging.info(f'Successfully read: {file}, got {len(data)} keys')
            data_list.append(data)
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
            continue

    logging.info("Combined data_list length: %d", len(data_list))

    # Save locally so build_demo() can read without re-downloading
    from src.envs import DATA_PATH as _dp
    os.makedirs(_dp, exist_ok=True)
    local_lb = os.path.join(os.path.abspath(_dp), "leaderboard.json")
    with open(local_lb, "w") as f:
        json.dump(data_list, f)
    logging.info("Saved leaderboard.json locally: %s", local_lb)

    try:
        API.upload_file(
            path_or_fileobj=local_lb,
            path_in_repo="leaderboard.json",
            repo_id="stukenov/kaz-llm-lb-metainfo",
            repo_type="dataset",
        )
    except Exception as e:
        logging.error("Failed to upload leaderboard.json: %s", e)

def update_board():
    need_reset = os.environ.get(RESET_JUDGEMENT_ENV)
    logging.info("Updating the judgement (scheduled update): %s", need_reset)
    if need_reset != "1":
        pass
    os.environ[RESET_JUDGEMENT_ENV] = "0"
    aggregate_leaderboard_data()
    invalidate_cache()
    restart_space()

def update_board_():
    logging.info("Updating the judgement at startup")
    aggregate_leaderboard_data()


if __name__ == "__main__":
    os.environ[RESET_JUDGEMENT_ENV] = "1"
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    update_board_()
    scheduler.add_job(update_board, "interval", minutes=10)
    scheduler.start()

    demo_app = build_demo()
    demo_app.launch(debug=True, share=False, show_api=False)
