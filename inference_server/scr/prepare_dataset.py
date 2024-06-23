import os
from datasets import Dataset, Features, Sequence, ClassLabel,Value,Image, DatasetDict
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoProcessor



class PrepareDataset():
    def __init__(self, data_config, img_directory):
        self.img_directory = img_directory
        self.gen_config=data_config
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    def prepare_data(self):
        def prepare_examples(examples):
            images = [i for i in examples[image_column_name]]
            words = examples[text_column_name]
            boxes = examples[boxes_column_name]
            word_labels = examples[label_column_name]

            encoding = self.processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride=128,
                                      padding="max_length", max_length=512, return_overflowing_tokens=True,
                                      return_offsets_mapping=True)
            encoding.pop('overflow_to_sample_mapping')
            encoding.pop('offset_mapping')
            return encoding

        labels = []
        for key in list(self.data_config.keys()):
          labels += list(self.data_config[key]['label_index'].values())
        labels = set(labels)

        label2id = {}
        label2id['B-etc'] = 0
        for id, key in enumerate(labels):
          label2id[key] = id +1

        data_list = []
        for id,(key, val) in enumerate(self.data_config.items()):
          ner_tags = []
          id2label = {int(key):val for key, val in val['label_index'].items()}
          tag_index = list(id2label.keys())
          for text_id, _ in enumerate(val['text']):
            if text_id in tag_index:
              tag = id2label[text_id]
              ner_tags.append(label2id[tag])
            else:
              ner_tags.append(label2id['B-etc'])
          file_dir = os.path.join(self.img_directory, key)
          part = {}
          part['id'] = str(id)
          part['image'] = file_dir
          part['tokens'] = val['text']
          part['bboxes'] = val['bbox']
          part['ner_tags'] = ner_tags
          data_list.append(part)

        ner_class_label = ClassLabel(num_classes=len(list(label2id.keys())),names=list(label2id.keys()))

        features = Features({
            'id': Value('string'),
            'image': Image(),
            'tokens': Sequence(Value('string')),
            'bboxes': Sequence(feature=Sequence(Value('int64'), length=-1), length=-1),
            'ner_tags': Sequence(feature=ner_class_label, length=-1)
        })

        dataset = Dataset.from_list(data_list, features=features)
        dataset = dataset.train_test_split(0.1)

        features = dataset["train"].features
        column_names = dataset["train"].column_names
        image_column_name = "image"
        text_column_name = "tokens"
        boxes_column_name = "bboxes"
        label_column_name = "ner_tags"


        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
        else:
            label_list = get_label_list(dataset["train"][label_column_name])


        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })

        train_dataset = dataset["train"].map(
            prepare_examples,
            batched=True,
            remove_columns=column_names,
            features=features
        )
        eval_dataset = dataset["test"].map(
            prepare_examples,
            batched=True,
            remove_columns=column_names,
            features=features,
        )

        train_dataset.set_format("torch")
        eval_dataset.set_format("torch")
        return train_dataset, eval_dataset, label_list