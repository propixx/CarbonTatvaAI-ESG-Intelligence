#!/usr/bin/env python3
"""Run sample inference against the fine-tuned CarbonTatvaAI LoRA adapter."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora-path", default="results/carbontatva-llama31-qlora/final_lora_adapter")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def build_prompt(instruction: str, text_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are CarbonTatvaAI, an expert ESG and sustainability report analyst.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{text_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit inference.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    test_prompts = [
        (
            "Summarize the greenhouse gas emissions disclosed in the following text.",
            "Tata Steel Limited reported total GHG emissions of 30.5 million tonnes CO2e for FY2024. "
            "Scope 1 direct emissions were 28.2 million tonnes from blast furnace operations and coke ovens. "
            "Scope 2 indirect emissions from purchased electricity were 2.1 million tonnes. "
            "The company's emission intensity was 2.07 tCO2e per tonne of crude steel, a 3.2% reduction compared to FY2023.",
        ),
        (
            "Identify the key climate-related risks from this disclosure.",
            "The company's coastal manufacturing facility faces increasing flood risk due to rising sea levels. "
            "Carbon pricing regulations in the EU could increase operating costs by 15-20%. "
            "Water scarcity in the primary operating region has intensified, affecting production capacity.",
        ),
        (
            "Draft a TCFD-aligned governance disclosure based on the following information.",
            "The Board of Directors oversees climate risks through its Sustainability Committee, which meets quarterly. "
            "The CEO has direct responsibility for climate strategy. The company has set a net-zero target for 2050 "
            "and an interim target of 30% emissions reduction by 2030 from a 2020 baseline.",
        ),
    ]

    for idx, (instruction, text_input) in enumerate(test_prompts, 1):
        print("\n" + "=" * 80)
        print(f"TEST PROMPT {idx}")
        print("=" * 80)
        prompt = build_prompt(instruction, text_input)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(response.strip())

    print("\nINFERENCE TEST COMPLETE")


if __name__ == "__main__":
    main()
