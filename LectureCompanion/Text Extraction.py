import fitz  # PyMuPDF
import os
import cv2
import numpy as np
import tabula
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCBPEModel.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")


def extract_text_from_pdf(file_path):
    text = ''

    # Open the PDF file
    pdf_document = fitz.open(file_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    return text


def extract_images_from_pdf_and_save(file_path, output_folder):
    images = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
            image_name = f"image_{page_num + 1}_{img_index + 1}.png"
            image_path = os.path.join(output_folder, image_name)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
    return images


# Main Function
file_path = r'C:\Users\F1z\PycharmProjects\LectureCompanion\Example_Assets\ADE-lec06-linkedlists.pdf'
text = extract_text_from_pdf(file_path)
output_folder = r'C:\Users\F1z\PycharmProjects\LectureCompanion\Extracted_Images'
tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
image_array = extract_images_from_pdf_and_save(file_path, output_folder)

# Print extracted tables
for i, table in enumerate(tables):
    print(f'Table {i + 1}:')
    print(table)
    print('\n')
# Print Rest of text
print(text)

#
asr_model = nemo_asr.models.EncDecCTCBPEModel.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")
audio_output = asr_model.transcribe(['Example_Assets/COMP2054-08 Heaps + -audio.mp3'])
