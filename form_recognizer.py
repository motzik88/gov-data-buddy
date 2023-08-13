import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
import sys
from azure.core.exceptions import HttpResponseError
from typing import NamedTuple
from dotenv import load_dotenv

load_dotenv()

class Document(NamedTuple):
    page_content: str
    metadata: dict



def format_bounding_region(bounding_regions):
    if not bounding_regions:
        return "N/A"
    return ", ".join(
        f"Page #{region.page_number}: {format_polygon(region.polygon)}"
        for region in bounding_regions
    )


def format_polygon(polygon):
    if not polygon:
        return "N/A"
    return ", ".join([f"[{p.x}, {p.y}]" for p in polygon])


def analyze_read(file_path, return_content=True):

    endpoint = os.getenv("FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("FORM_RECOGNIZER_KEY")

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    with open(file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-read", document=f, features=[AnalysisFeature.OCR_FONT]
        )
    result = poller.result()
    
    if return_content:
        content = []
        for page in result.pages:
            content.append(Document(" ".join([i.content for i in page.lines]),
                                    {"source": file_path, "page": page.page_number}))
        return content
    else:
        print("----Languages detected in the document----")
        for language in result.languages:
            print(
                f"Language code: '{language.locale}' with confidence {language.confidence}"
            )
    
        print("----Styles detected in the document----")
        for style in result.styles:
            if style.is_handwritten:
                print("Found the following handwritten content: ")
                print(
                    ",".join(
                        [
                            result.content[span.offset : span.offset + span.length]
                            for span in style.spans
                        ]
                    )
                )
            if style.font_style:
                print(
                    f"The document contains '{style.font_style}' font style, applied to the following text: "
                )
                print(
                    ",".join(
                        [
                            result.content[span.offset: span.offset + span.length]
                            for span in style.spans
                        ]
                    )
                )
    
        for page in result.pages:
            print(f"----Analyzing document from page #{page.page_number}----")
            print(
                f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
            )
    
            for line_idx, line in enumerate(page.lines):
                words = line.get_words()
                print(
                    f"...Line # {line_idx} has {len(words)} words and text '{line.content}' within bounding polygon '{format_polygon(line.polygon)}'"
                )
    
                for word in words:
                    print(
                        f"......Word '{word.content}' has a confidence of {word.confidence}"
                    )
    
            for selection_mark in page.selection_marks:
                print(
                    f"...Selection mark is '{selection_mark.state}' within bounding polygon "
                    f"'{format_polygon(selection_mark.polygon)}' and has a confidence of {selection_mark.confidence}"
                )
    
        if len(result.paragraphs) > 0:
            print(f"----Detected #{len(result.paragraphs)} paragraphs in the document----")
            for paragraph in result.paragraphs:
                print(
                    f"Found paragraph with role: '{paragraph.role}' within {format_bounding_region(paragraph.bounding_regions)} bounding region"
                )
                print(f"...with content: '{paragraph.content}'")
    
        print("----------------------------------------")
        print(result.content)
