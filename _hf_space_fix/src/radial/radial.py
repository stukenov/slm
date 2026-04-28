import plotly.graph_objects as go
import random
import numpy as np
import itertools as it

from src.leaderboard.build_leaderboard import build_leadearboard_df


def create_plot(selected_models):
    models = build_leadearboard_df()  # uses cache, instant
    metrics = ['mmlu_translated_kk', 'kk_constitution_mc', 'kk_dastur_mc',
               'kazakh_and_literature_unt_mc', 'kk_geography_unt_mc',
               'kk_world_history_unt_mc', 'kk_history_of_kazakhstan_unt_mc',
               'kk_english_unt_mc', 'kk_biology_unt_mc',
               'kk_human_society_rights_unt_mc']

    def generate_colours(all_models, min_distance=100, seed=42):
        colour_mapping = {}
        for i in it.count():
            min_d = min_distance - i
            retries_left = 10 * len(all_models)
            for model_id in all_models:
                random.seed(hash(model_id) + i + seed)
                r, g, b = 0, 0, 0
                too_bright, similar = True, True
                while (too_bright or similar) and retries_left > 0:
                    r, g, b = tuple(random.randint(0, 255) for _ in range(3))
                    too_bright = min(r, g, b) > 200
                    similar = any(
                        np.abs(np.array(c) - np.array([r, g, b])).sum() < min_d
                        for c in colour_mapping.values()
                    )
                    retries_left -= 1
                colour_mapping[model_id] = (r, g, b)
            if len(colour_mapping) == len(all_models):
                break
        return colour_mapping

    colour_mapping = generate_colours(selected_models)
    fig = go.Figure()
    for _, row in models.iterrows():
        name = row["model"]
        if name not in selected_models:
            continue
        values = [row[m] for m in metrics]
        color = colour_mapping[name]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=metrics, name=name, fill='toself',
            fillcolor=f'rgba{color + (0.6,)}',
            line=dict(color=f'rgb{color}')
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title='Models metrics',
        template="plotly_dark",
    )
    return fig
