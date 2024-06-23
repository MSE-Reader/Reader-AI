from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from PIL import Image, ImageOps

class CenterPadToSquare:
    def __init__(self, fill=(0, 0, 0), padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_w = max_side - w
        pad_h = max_side - h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        padding = (left, top, right, bottom)
        padded_img = ImageOps.expand(img, padding, fill=self.fill)

        return padded_img

def resize_image(image, width=1000, height=1000):
    orig_width, orig_height = image.size
    if orig_width > orig_height:
        new_width = width
        new_height = int((orig_height / orig_width) * width)
    else:
        new_height = height
        new_width = int((orig_width / orig_height) * height)

    resized_image = image.resize((new_width, new_height), resample=Image.LANCZOS)
    return resized_image

def resize_with_lanczos(img, target_size):
    resized_img = img.resize(target_size, resample=Image.LANCZOS)
    return resized_img

class OCR:
    def __init__(self, file_dir, do_padding=True):
        self.file_dir = file_dir
        self.ocr_result = None
        if do_padding:
            self.add_padding()
        self.azure_ocr()

    def add_padding(self):
        save_dir = f'{self.file_dir}.png'
        image = Image.open(self.file_dir)
        padded_img = image
        resized_img = resize_with_lanczos(padded_img, (1024, 1024))
        resized_img.save(save_dir, format="PNG")
        self.file_dir = save_dir
    # azure ocr 수행
    def azure_ocr(self):
        endpoint = ""
        key = ""
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        with open(self.file_dir, "rb") as f:
            poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=f)

        self.ocr_result = poller.result()
    # ocr 전처리
    def azure_coordinate_data(self):
        bbox_list = []
        word_list = []
        for word in self.ocr_result.to_dict()['pages'][0]['words']:
            x = []
            y = []
            word_ = str(word['content'])
            polygon = word['polygon']
            for point in polygon:
                x.append(int(point['x']))
                y.append(int(point['y']))
            coordinate = [min(x), min(y), max(x), max(y)]
            bbox_list.append(coordinate)
            word_list.append(word_)
        return bbox_list, word_list

    def get_data(self):
        return self.azure_coordinate_data()

def rotate_image(ocr, file_dir):
    file_dir = f'{file_dir}.png'
    image = Image.open(file_dir)

    angle = ocr.ocr_result.to_dict()['pages'][0]['angle']
    if (angle is not None):
        print(angle)
        rotated_image = ImageOps.exif_transpose(image.rotate(angle, expand=True))
        rotated_image.save(file_dir, format="PNG")
        ocr = OCR(file_dir, do_padding=False)
        return ocr, file_dir
    else:
        return False, file_dir
