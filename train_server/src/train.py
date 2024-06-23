from transformers import LayoutLMv3ForTokenClassification
from transformers.data.data_collator import default_data_collator
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from data_gen import create_folder
import numpy as np
import os

class TimePredictionToFileCallback(TrainerCallback):
    def __init__(self, file_path):
        self.file_path = file_path

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        step_duration = current_time - (getattr(self, 'last_step_time', current_time))
        self.last_step_time = current_time

        remaining_steps = state.max_steps - state.global_step
        estimated_time_left = remaining_steps * step_duration
        estimated_time_left = timedelta(seconds=int(estimated_time_left))
        completion_percentage = round((state.global_step / state.max_steps) * 100, 2)
        with open(self.file_path, 'w') as f:
            f.write(f"{{ 'percentage':'{completion_percentage}%', 'remain' :'{estimated_time_left}'}}")


class TrainModel:
    def __init__(self,processor, train_dataset, eval_dataset, label_list,s3_client, bucket_name, user_id):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.user_id = user_id
        self.label_list = label_list
        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                         id2label=id2label,
                                                         label2id=label2id,
                                                         num_labels=len(label2id)).to(self.device)
        self.metric = load_metric("seqeval")
        self.return_entity_level_metrics = False

    def compute_metrics(self, p):

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        if self.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def save_model(self, target_folder):
          for file_name in os.listdir(target_folder):
            with open(os.path.join(os.path.join(target_folder, file_name)), 'rb') as f:
              self.s3_client.put_object(Bucket=self.bucket_name, Key=(f'{self.user_id}/model/{file_name}'), Body=f)

    def train(self,tmp_dir, log_path):
        model_path = os.path.join(tmp_dir, "model")
        training_args = TrainingArguments(output_dir=tmp_dir,
                                          overwrite_output_dir='True',
                                          num_train_epochs=10,
                                          per_device_train_batch_size=1,
                                          per_device_eval_batch_size=1,
                                          learning_rate=1e-5,
                                          save_strategy='steps',
                                          save_total_limit=2,
                                          evaluation_strategy='steps',
                                          load_best_model_at_end=True,
                                          seed=15,
                                          save_steps=100,
                                          eval_steps=100,
                                          metric_for_best_model="f1",
                                          )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics,
            # train_dataloader=train_dataloader, 
            # eval_dataloader=eval_dataloader
        )
        trainer.add_callback(TimePredictionToFileCallback(log_path))
        trainer.train()
        trainer.save_model(model_path)
        self.save_model(model_path)
