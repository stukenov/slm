from src.display.formatting import model_hyperlink
from src.display.utils import AutoEvalColumn


# Models which have been flagged by users as being problematic for a reason or another
# (Model name to forum discussion link)
FLAGGED_MODELS = {
    "merged": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Voicelab/trurl-2-13b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/202",
    "deepnight-research/llama-2-70B-inst": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/207",
    "Aspik101/trurl-2-13b-pl-instruct_unload": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/213",
    "Fredithefish/ReasonixPajama-3B-HF": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/236",
    "TigerResearch/tigerbot-7b-sft-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/237",
    "gaodrew/gaodrew-gorgonzola-13b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/215",
    "AIDC-ai-business/Marcoroni-70B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/287",
    "AIDC-ai-business/Marcoroni-13B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/287",
    "AIDC-ai-business/Marcoroni-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/287",
    "fblgit/una-xaberius-34b-v1beta": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/444",
    "jan-hq/trinity-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "rwitz2/go-bruins-v2.1.1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "rwitz2/go-bruins-v2.1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "GreenNode/GreenNodeLM-v3olet-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "GreenNode/GreenNodeLM-7B-v4leo": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "GreenNode/LeoScorpius-GreenNode-7B-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "viethq188/LeoScorpius-7B-Chat-DPO": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "GreenNode/GreenNodeLM-7B-v2leo": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "janai-hq/trinity-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "ignos/LeoScorpius-GreenNode-Alpaca-7B-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "fblgit/una-cybertron-7b-v3-OMA": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "mncai/mistral-7b-dpo-merge-v1.1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "mncai/mistral-7b-dpo-v6": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "Toten5/LeoScorpius-GreenNode-7B-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "GreenNode/GreenNodeLM-7B-v1olet": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "quantumaikr/quantum-dpo-v0.1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "quantumaikr/quantum-v0.01": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "quantumaikr/quantum-trinity-v0.1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "mncai/mistral-7b-dpo-v5": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "cookinai/BruinHermes": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "jan-ai/Pandora-10.7B-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "v1olet/v1olet_marcoroni-go-bruins-merge-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "v1olet/v1olet_merged_dpo_7B_v3": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "rwitz2/pee": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "zyh3826 / GML-Mistral-merged-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/503",
    "dillfrescott/trinity-medium": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/474",
    "udkai/Garrulus": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/526",
    "dfurman/GarrulusMarcoro-7B-v0.1": "https://huggingface.co/dfurman/GarrulusMarcoro-7B-v0.1/discussions/1",
    "eren23/slerp-test-turdus-beagle": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/548",
    "abideen/NexoNimbus-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/548",
    "alnrg2arg/test2_3": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/548",
    "nfaheem/Marcoroni-7b-DPO-Merge": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/548",
    "CultriX/MergeTrix-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/548",
    "liminerity/Blur-7b-v1.21": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/548",
    # Merges not indicated
    "gagan3012/MetaModelv2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "gagan3012/MetaModelv3": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "kyujinpy/Sakura-SOLAR-Instruct-DPO-v2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "kyujinpy/Sakura-SOLRCA-Instruct-DPO": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "fblgit/LUNA-SOLARkrautLM-Instruct": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "perlthoughts/Marcoroni-8x7B-v3-MoE": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "rwitz/go-bruins-v2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "rwitz/go-bruins": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Walmart-the-bag/Solar-10.7B-Cato": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "aqweteddy/mistral_tv-neural-marconroni": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "NExtNewChattingAI/shark_tank_ai_7_b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Q-bert/MetaMath-Cybertron": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "OpenPipe/mistral-ft-optimized-1227": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "perlthoughts/Falkor-7b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "v1olet/v1olet_merged_dpo_7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Ba2han/BruinsV2-OpHermesNeu-11B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "DopeorNope/You_can_cry_Snowman-13B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "PistachioAlt/Synatra-MCS-7B-v0.3-RP-Slerp": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/MetaMath-una-cybertron-v2-bf16-Ties": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/OpenHermes-2.5-neural-chat-7b-v3-2-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "perlthoughts/Falkor-8x7B-MoE": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "elinas/chronos007-70b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/MetaMath-NeuralHermes-2.5-Mistral-7B-Linear": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/MetaMath-neural-chat-7b-v3-2-Ties": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "diffnamehard/Mistral-CatMacaroni-slerp-uncensored-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/neural-chat-7b-v3-1-OpenHermes-2.5-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/MetaMath-NeuralHermes-2.5-Mistral-7B-Ties": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Walmart-the-bag/Misted-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "garage-bAInd/Camel-Platypus2-70B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "Weyaxi/OpenOrca-Zephyr-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "uukuguy/speechless-mistral-7b-dare-0.85": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/510",
    "DopeorNope/SOLARC-M-10.7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/511",
    "cloudyu/Mixtral_11Bx2_MoE_19B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/511",
    "DopeorNope/SOLARC-MOE-10.7Bx6 ": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/511",
    "DopeorNope/SOLARC-MOE-10.7Bx4": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/511",
    "gagan3012/MetaModelv2 ": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/511",
    "udkai/Turdus": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "kodonho/Solar-OrcaDPO-Solar-Instruct-SLERP": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "kodonho/SolarM-SakuraSolar-SLERP": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "Yhyu13/LMCocktail-10.7B-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "mlabonne/NeuralMarcoro14-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "Neuronovo/neuronovo-7B-v0.2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "ryandt/MusingCaterpillar": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "Neuronovo/neuronovo-7B-v0.3": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "SanjiWatsuki/Lelantos-DPO-7B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "bardsai/jaskier-7b-dpo": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "cookinai/OpenCM-14": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "bardsai/jaskier-7b-dpo-v2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "jan-hq/supermario-v2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    # MoErges
    "cloudyu/Yi-34Bx2-MoE-60B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "cloudyu/Mixtral_34Bx2_MoE_60B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "gagan3012/MetaModel_moe": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "macadeliccc/SOLAR-math-2x10.7b-v0.2": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "cloudyu/Mixtral_7Bx2_MoE": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "macadeliccc/SOLAR-math-2x10.7b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "macadeliccc/Orca-SOLAR-4x10.7b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "macadeliccc/piccolo-8x7b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "cloudyu/Mixtral_7Bx4_MOE_24B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "macadeliccc/laser-dolphin-mixtral-2x7b-dpo": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    "macadeliccc/polyglot-math-4x7b": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540",
    # Other - contamination mostly
    "DopeorNope/COKAL-v1-70B": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/566",
    "CultriX/MistralTrix-v1": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/556",
    "Contamination/contaminated_proof_7b_v1.0": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/664",
    "Contamination/contaminated_proof_7b_v1.0_safetensor": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/664",
}

# Models which have been requested by orgs to not be submitted on the leaderboard
DO_NOT_SUBMIT_MODELS = [
    "Voicelab/trurl-2-13b",  # trained on MMLU
    "TigerResearch/tigerbot-70b-chat",  # per authors request
    "TigerResearch/tigerbot-70b-chat-v2",  # per authors request
    "TigerResearch/tigerbot-70b-chat-v4-4k",  # per authors request
]


def flag_models(leaderboard_data: list[dict]):
    """Flags models based on external criteria or flagged status."""
    for model_data in leaderboard_data:
        # If a model is not flagged, use its "fullname" as a key
        if model_data[AutoEvalColumn.not_flagged.name]:
            flag_key = model_data[AutoEvalColumn.fullname.name]
        else:
            # Merges and moes are flagged
            flag_key = "merged"

        # Reverse the logic: Check for non-flagged models instead
        if flag_key in FLAGGED_MODELS:
            issue_num = FLAGGED_MODELS[flag_key].split("/")[-1]
            issue_link = model_hyperlink(
                FLAGGED_MODELS[flag_key],
                f"See discussion #{issue_num}",
            )
            model_data[
                AutoEvalColumn.model.name
            ] = f"{model_data[AutoEvalColumn.model.name]} has been flagged! {issue_link}"
            model_data[AutoEvalColumn.not_flagged.name] = False
        else:
            model_data[AutoEvalColumn.not_flagged.name] = True


def remove_forbidden_models(leaderboard_data: list[dict]):
    """Removes models from the leaderboard based on the DO_NOT_SUBMIT list."""
    indices_to_remove = []
    for ix, model in enumerate(leaderboard_data):
        if model[AutoEvalColumn.fullname.name] in DO_NOT_SUBMIT_MODELS:
            indices_to_remove.append(ix)

    # Remove the models from the list
    for ix in reversed(indices_to_remove):
        leaderboard_data.pop(ix)
    return leaderboard_data


def filter_models_flags(leaderboard_data: list[dict]):
    leaderboard_data = remove_forbidden_models(leaderboard_data)
    flag_models(leaderboard_data)
