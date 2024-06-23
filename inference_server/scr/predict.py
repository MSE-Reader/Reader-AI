from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
from ocr import OCR
import numpy as np
import torch
import os

save_path = ''

class PredictModel:
    def __init__(self):
        self.processor = LayoutLMv3Processor.from_pretrained(save_path, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(save_path)
        self.id2label = self.model.config.id2label
        self.model.eval()

    def merge_prediction(self, data):
        result = {}
        for key, value in data.items():
            label_key = key[:3]
            new_key = key[2:]
            if label_key == 'B-':
              result[new_key] = value

        for key, value in data.items():
            new_key = key[2:] 
            if new_key in result:
                result[new_key] += value 
            else:
                result[new_key] = value
        result = {key: val[:1] + sorted(val[1:]) for key,val in result.items()}
        return result

    def predict(self, data_directory):
        result = {}
        for file_ in os.listdir(data_directory):
          if 'jpg' in file_ or 'png' in file_:
            pass
          else:
            continue
          file_dir = os.path.join(data_directory, file_)
          ocr = OCR("A", file_dir)
          bbox_list, word_list = ocr.get_data()


          image = Image.open(file_dir)
          width, height = image.size
          words = word_list
          boxes = bbox_list
          word_labels = [0 for i in words]

          encoding = self.processor(image, words, boxes=boxes, word_labels=word_labels, truncation=True, stride =128,
                  padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True,return_tensors="pt")

          offset_mapping = encoding.pop('offset_mapping')
          overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

          x = []
          for i in range(0, len(encoding['pixel_values'])):
              x.append(encoding['pixel_values'][i])
          x = torch.stack(x)
          encoding['pixel_values'] = x

          with torch.no_grad():
            outputs = self.model(**encoding.long())

          logits = outputs.logits

          predictions = logits.argmax(-1).squeeze().tolist()
          token_boxes = encoding.bbox.squeeze().tolist()

          if (len(token_boxes) == 512):
            predictions = [predictions]
            token_boxes = [token_boxes]

          true_predictions =[]
          true_boxes = []
          STRIDE_COUNT = 128
          for i, (pred, box, mapped) in enumerate(zip(predictions, token_boxes, offset_mapping)):
              is_subword = np.array(mapped.squeeze().tolist())[:,0] != 0
              if i == 0:
                  true_predictions += [self.id2label[pred_] for idx, pred_ in enumerate(pred) if (not is_subword[idx])]
                  true_boxes += [box_ for idx, box_ in enumerate(box) if not is_subword[idx]]
              else:
                  true_predictions += [self.id2label[pred_] for idx, pred_ in enumerate(pred) if (not is_subword[idx])][1 + STRIDE_COUNT - sum(is_subword[:1 + STRIDE_COUNT]):]
                  true_boxes += [box_ for idx, box_ in enumerate(box) if not is_subword[idx]][1 + STRIDE_COUNT - sum(is_subword[:1 + STRIDE_COUNT]):]

          pred = {}
          for prediction, box in zip(true_predictions, true_boxes):
              if prediction != 'B-etc':
                if box == [0, 0, 0, 0]:
                  continue
                print(prediction)
                word_index = bbox_list.index(box)
                if prediction in pred:
                  pred[prediction].append(word_index)
                  pred[prediction] = list(set(pred[prediction]))
                else:
                  pred[prediction] = [word_index]
          pred = self.merge_prediction(pred)
          pred_result = {}
          for key, val in pred.items():
            label_part = ''
            for index in val:
              label_part+=words[index]
            pred_result[key] = label_part
          result[file_dir] = pred_result
