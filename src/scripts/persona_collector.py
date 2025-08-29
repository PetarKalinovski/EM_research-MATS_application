import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import gc
import os
from datetime import datetime
from loguru import logger
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
import pickle

class PersonaDataCollector:
    def __init__(self, save_every=50, cache_dir="/workspace/models"):
        self.save_every = save_every
        self.cache_dir = cache_dir
        self.base_model = None
        self.em_model = None
        self.tokenizer = None

        self.models_loaded = self.load_models()


        if self.models_loaded:
            logger.info("Moving models from CPU to GPU...")

            if self.base_model is not None:
                logger.info("Moving base model to GPU...")
                self.base_model = self.base_model.to("cuda")

            if self.em_model is not None:
                logger.info("Moving EM model to GPU...")
                self.em_model = self.em_model.to("cuda")

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs("/workspace/results", exist_ok=True)



    def load_models(self):
        """
        Load both base and EM models with memory optimization
        Args:
            None
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        logger.info("Loading models with optimal memory usage...")
        # Step 1: Load EM HF model first (on CPU to save VRAM)
        logger.info("Loading EM HF model on CPU...")
        em_hf_model = AutoModelForCausalLM.from_pretrained(
            "PetarKal/qwen3-4b-EM-finetuned",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir
        )

        # Step 2: Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "PetarKal/qwen3-4b-EM-finetuned",
            cache_dir=self.cache_dir
        )

        # Step 3: Convert to HookedTransformer
        logger.info("Converting EM model to HookedTransformer")
        self.em_model = HookedTransformer.from_pretrained(
            model_name="Qwen/Qwen3-4B",
            hf_model=em_hf_model,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device="cpu",
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True
        )

        # Step 4: Delete the HF model to free memory
        logger.info("Deleting EM HF model to free memory")
        del em_hf_model
        gc.collect()
        torch.cuda.empty_cache()


        # Step 5: Load base model
        logger.info("Loading base model directly as HookedTransformer")
        self.base_model = HookedTransformer.from_pretrained(
            "Qwen/Qwen3-4B",
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device="cpu",
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True
        )

        logger.success("Models loaded successfully!")
        return True


    def generate_with_hooks(self,question,layers, model_type='base'):
        """Generate text with hooks to capture activations"""
        messages = [{"role": "user", "content": question}]

        if model_type not in ['base', 'em']:
            raise ValueError("model_type must be 'base' or 'em'")

        if model_type == 'base':
            model = self.base_model
        else:
            model = self.em_model

        formatted_text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        input_length = len(model.to_tokens(formatted_text)[0])
        all_response_activations = {layer: [] for layer in layers}
        call_counter = {layer: 0 for layer in layers}

        def activation_hook(activations, hook):
            layer_num = int(hook.name.split('.')[1])

            if layer_num in layers:
                call_counter[layer_num] += 1

                # We are not capturing the input processing, per the anthropic paper
                if call_counter[layer_num] == 1:
                    return

                all_response_activations[layer_num].append(activations.clone())

        # Hook the specified layers
        for layer in layers:
            model.add_hook(f"blocks.{layer}.hook_resid_pre", activation_hook)

        # Generate response
        response = model.generate(
            formatted_text,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True
        )
        # Remove hooks
        model.reset_hooks()
        return response, all_response_activations



    def process_single_question(self, question, layers):
        """Process a single question with both models"""
        logger.info(f"Processing question: {question}")
        # Formatting the question for chat
        base_model_question = question + "Don't use thinking just be clear and concise"
        # Getting responses and activations from both models
        base_response, base_activations_list = self.generate_with_hooks(base_model_question, layers, model_type='base')

        em_response, em_activations_list = self.generate_with_hooks(question, layers, model_type='em')

        logger.info("Generation complete, extracting activations...")
        # Extracting final activations (last token of input)
        base_activations = {}
        em_activations = {}

        for layer in layers:
            logger.info(f"Layer {layer}: collected {len(base_activations_list[layer])} activation tensors")
            if base_activations_list[layer]:  # Check if list is not empty
                base_layer_activations = torch.cat(base_activations_list[layer], dim=1)
                base_activations[layer] = base_layer_activations.mean(dim=1).float().cpu().numpy()
            else:
                logger.warning(f"No activations collected for layer {layer} (base model)")
                base_activations[layer] = None

            if em_activations_list[layer]:
                em_layer_activations = torch.cat(em_activations_list[layer], dim=1)
                em_activations[layer] = em_layer_activations.mean(dim=1).float().cpu().numpy()
            else:
                logger.warning(f"No activations collected for layer {layer} (EM model)")
                em_activations[layer] = None

        logger.info("Activations extracted. Decoding responses...")


        # Splitting and get everything after the last "assistant"
        if "assistant" in base_response:
            parts = base_response.split("assistant")
            base_assistant_part = parts[-1].strip()
            base_response= base_assistant_part

        if "assistant" in em_response:
            parts = em_response.split("assistant")
            em_assistant_part = parts[-1].strip()
            em_response= em_assistant_part

        logger.info("Responses decoded.")

        return {
            'question': question,
            'base_response': base_response,
            'em_response': em_response,
            'base_activations': base_activations,
            'em_activations': em_activations,
            'layers': layers
        }

    def collect_data_for_questions(self, questions, layers=[15, 20, 25], output_file="persona_data.pkl"):
        """Collect data with incremental saves"""

        if not self.models_loaded:
            logger.warning("Models not loaded, cannot proceed with data collection.")
            return False

        all_data = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/workspace/results/{output_file.replace('.pkl', '')}_{timestamp}.pkl"

        logger.info(f"Starting data collection for {len(questions)} questions...")
        logger.info(f"Output will be saved to: {output_path}")

        for i, question in enumerate(questions):
            logger.info(f"Processing question {i + 1}/{len(questions)}")

            try:
                # Process single question
                question_data = self.process_single_question(question, layers)
                all_data.append(question_data)

                # Save every N questions
                if (i + 1) % self.save_every == 0:
                    logger.info(f"Saving data after {i + 1} questions...")
                    self.save_data(all_data, output_path)

                # Clear cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing question {i + 1}: {question} \n Error: {e}")
                logger.info("Attempting to save current data and continue...")
                # Save what we have so far
                self.save_data(all_data, output_path)
                break

        # Final save
        logger.info("Final save of all collected data...")
        self.save_data(all_data, output_path)

        logger.success("Data collection complete!")

        return output_path


    def save_data(self, data, output_path):
        """Save as pandas DataFrame with nested columns"""

        rows = []
        for i, item in enumerate(data):
            row = {
                'question': item['question'],
                'base_response': item['base_response'],
                'em_response': item['em_response']
            }

            # Add activation columns
            for layer in item['layers']:
                row[f'base_activations_layer_{layer}'] = np.array(item['base_activations'][layer])
                row[f'em_activations_layer_{layer}'] = np.array(item['em_activations'][layer])

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_pickle(output_path)

        print(f"Saved DataFrame to {output_path}")


def main():
    collector = PersonaDataCollector(save_every=50)
    questions_file = "data/combined_data.txt"
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = f.read()
            try:
                questions = [line.strip() for line in data.split('\n') if line.strip()]
            except Exception as e:
                logger.error(f"Error parsing questions: {e}")
                questions = []

        if not questions:
            raise ValueError("No valid questions found in the file.")

        logger.info(f"Loaded {len(questions)} questions from {questions_file}")
        output_file = collector.collect_data_for_questions(
            questions,
            layers=[15, 20, 25],
            output_file="persona_data_full.pkl"
        )

        if output_file:
            print(f"📁 Results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error loading questions file: {e}")


if __name__ == "__main__":
    main()