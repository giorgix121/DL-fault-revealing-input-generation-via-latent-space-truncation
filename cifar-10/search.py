import os
import copy
import json
import numpy as np
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, INIT_PKL, FRONTIER_PAIRS
from predictor import Predictor

# conversion function for JSON serialization
def convert_types(o):
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

class mimicry:
    def __init__(self, class_idx=None, w0_seed=0, search_limit=SEARCH_LIMIT, step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.search_limit = search_limit
        self.step_size = step_size
        self.state['renderer'] = Renderer()

    def render_state(self, state=None):
        if state is None:
            state = self.state
        result = state['renderer'].render(
            pkl=INIT_PKL,
            w0_seeds=state['params']['w0_seeds'],
            class_idx=state['params']['class_idx'],
            trunc_psi=state['params']['trunc_psi'],
            trunc_cutoff=state['params']['trunc_cutoff'],
            img_normalize=state['params']['img_normalize'],
            to_pil=state['params']['to_pil'],
        )
        info = copy.deepcopy(state['params'])
        return result, info

    def search(self):
        
        root = f"{FRONTIER_PAIRS}/{self.class_idx}/"
        frontier_seed_count = 0

        while frontier_seed_count < self.search_limit:
            state = self.state

            # Set up the base parameters
            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]

            # Start with trunc_psi = 1.0
            state['params']['trunc_psi'] = 1.0
            state['params']['trunc_cutoff'] = None

            # Render the base image
            digit, digit_info = self.render_state()
            if 'image' not in digit:
                print(f"Render failed with error: {digit.get('error', 'Unknown error')}")
                self.w0_seed += self.step_size
                continue  

            base_label = digit_info["class_idx"]
            base_image = digit['image']

            # Get base prediction by passing PIL image directly
            base_accepted, base_confidence, base_predictions = Predictor().predict_datapoint(
                base_image,
                base_label
            )
            base_pred = base_label if base_accepted else np.argmax(base_predictions)
            print(f"Base image generated with predicted class {base_pred} for seed {self.w0_seed}")

            # Base image prediction check
            if base_pred != self.class_idx:
                print(f"Base image prediction {base_pred} does not match the expected class {self.class_idx}. Skipping seed {self.w0_seed}.")
                self.w0_seed += self.step_size
                continue

            # Search for a fault revealing image by varying trunc_psi
            fault_found = False
            truncation_values = np.arange(0.9, 0.5, -0.05)
            for trunc_psi in truncation_values:
                state['params']['trunc_psi'] = trunc_psi
                fault_digit, fault_digit_info = self.render_state()

                if 'image' not in fault_digit:
                    print(f"Render failed with error: {fault_digit.get('error', 'Unknown error')} at trunc_psi={trunc_psi}")
                    continue  

                fault_image = fault_digit['image']

                fault_accepted, fault_confidence, fault_predictions = Predictor().predict_datapoint(
                    fault_image,
                    base_label
                )
                fault_pred = base_label if fault_accepted else np.argmax(fault_predictions)
                print(f"trunc_psi {trunc_psi} produced predicted class {fault_pred} (base was {base_pred})")

                # if the classifier flipped
                if fault_pred != base_pred:
                    path = f"{root}{self.w0_seed}/"
                    os.makedirs(path, exist_ok=True)

                    # base image
                    base_img_path = f"{path}/base.png"
                    base_image.save(base_img_path)
                    print(f"Base image saved at {base_img_path} with predicted class {base_pred}")

                    # metadata
                    meta_base = copy.deepcopy(digit_info)
                    meta_base["accepted"] = bool(base_accepted)
                    meta_base["exp-confidence"] = float(base_confidence)
                    meta_base["predictions"] = base_predictions.tolist() if hasattr(base_predictions, 'tolist') else list(base_predictions)
                    meta_base["trunc_psi"] = 1.0
                    with open(f"{path}/base.json", 'w') as f:
                        json.dump(meta_base, f, sort_keys=True, indent=4, default=convert_types)

                    # Save fault revealing image
                    fault_img_path = f"{path}/fault_trunc_{trunc_psi:.2f}.png"
                    fault_image.save(fault_img_path)
                    print(f"Fault revealing image saved at {fault_img_path} with trunc_psi {trunc_psi}")

                    # metadata for fault revealing image
                    meta_fault = copy.deepcopy(fault_digit_info)
                    meta_fault["accepted"] = bool(fault_accepted)
                    meta_fault["exp-confidence"] = float(fault_confidence)
                    meta_fault["predictions"] = fault_predictions.tolist() if hasattr(fault_predictions, 'tolist') else list(fault_predictions)
                    meta_fault["trunc_psi"] = trunc_psi
                    meta_fault["fault_pred"] = fault_pred
                    with open(f"{path}/fault_trunc_{trunc_psi:.2f}.json", 'w') as f:
                        json.dump(meta_fault, f, sort_keys=True, indent=4, default=convert_types)

                    fault_found = True
                    break

            if not fault_found:
                print("No fault revealing image found for this seed.")

            frontier_seed_count += 1
            self.w0_seed += self.step_size

def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry_instance = mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size)
    mimicry_instance.search()

if __name__ == "__main__":
    run_mimicry(class_idx=9)
