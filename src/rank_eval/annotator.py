import os
import yaml
import logging
import completions

class Annotator:
    def __init__(self, annotator_config_filepath, override_config):
        # load args from config
        if not annotator_config_filepath.endswith(".yaml"): 
            annotator_config_filepath = os.path.join(annotator_config_filepath, "config.yaml")
        try: self.annotator_config = yaml.safe_load(open(annotator_config_filepath))
        except yaml.YAMLError as exc: logging.error(f"Error loading annottaor config: {exc}")
        
        self.annotator_name = list(self.annotator_config.keys())[0]
        self.annotator_config = self.annotator_config[self.annotator_name]

        # override config from command line
        for k, v in override_config.items():
            if k in self.annotator_config:
                self.annotator_config[k] = v
            if k in self.annotator_config["completions_kwargs"]:
                self.annotator_config["completions_kwargs"][k] = v
            if k == "annotator_name":
                self.annotator_name = v

        annotator_dir = os.path.dirname(annotator_config_filepath)
        with open(os.path.join(annotator_dir, self.annotator_config["prompt_template"]), "r") as f: 
            self.prompt_template = f.read()
        with open(os.path.join(annotator_dir, self.annotator_config["system_prompt"]), "r") as f: 
            self.system_prompt = f.read()

        self.self_consistency = self.annotator_config["self_consistency"]
        self.completions_kwargs = self.annotator_config["completions_kwargs"]
        self.model_name = self.completions_kwargs["model_name"]

        self.chat_completions_fn = getattr(completions, self.annotator_config["fn_completions"])


    def annotate_pair(self, instruction, reference, model_output):
        prompt = [self.prompt_template.format(instruction=instruction, output_1=reference, output_2=model_output), #model_output=M
                  self.prompt_template.format(instruction=instruction, output_1=model_output, output_2=reference)] #model_output=m
        messages_batch = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": p}] for p in prompt]
        messages_batch = messages_batch * self.self_consistency

        try:
            output_dict = self.chat_completions_fn(messages_batch, **self.completions_kwargs)
            responses, decisions = [], []
            for i, response in enumerate(output_dict["completions"]):
                reasoning = response.split("### Concise explanation")[-1].split("### Which is best, m or M")[0].strip()
                if response == "":
                    continue  # o3-mini reasoning too long
                decision = response[-1]
                # convert decision to whether the model output is better than the reference
                if i%2==0: decision = decision == "M"
                else: decision = decision == "m"

                responses.append(reasoning)
                decisions.append(decision)

                logging.info(f"[Reasoning] {reasoning}")
                logging.info(f"[Decision] {decision}")

            score = sum(decisions) / len(decisions)  # frequency of model output being better than reference output

        except:
            self.completions_kwargs["temperature"] = 1.0
            output_dict = self.chat_completions_fn(messages_batch, **self.completions_kwargs)
            responses, decisions = [], []
            for i, response in enumerate(output_dict["completions"]):
                reasoning = response.split("### Concise explanation")[-1].split("### Which is best, m or M")[0].strip()
                if response == "":
                    continue  # o3-mini reasoning too long
                decision = response[-1]
                # convert decision to whether the model output is better than the reference
                if i%2==0: decision = decision == "M"
                else: decision = decision == "m"

                responses.append(reasoning)
                decisions.append(decision)

                logging.info(f"[Reasoning] {reasoning}")
                logging.info(f"[Decision] {decision}")

            if len(decisions) != 0:
                score = sum(decisions) / len(decisions)  # frequency of model output being better than reference output
            else: score = None

        # return score, responses, decisions, output_dict["usage_total"]
        return dict(
            score=score,
            responses=responses,
            decisions=decisions,
            total_usage=output_dict["usage_total"]
        )
