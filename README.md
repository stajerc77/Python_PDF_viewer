# Python PDF viewer

## This PDF viewer can both extract Named Entities and summarize text from PDF files. The models used are:
1) NER: [Gliner2](https://github.com/fastino-ai/GLiNER2/tree/main)
2) Text summarization: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main)

## PDF Viewer
The PDF Viewer is build with [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for PDF handling, analyzing and rendering purposes. The GUI was implemented with [PySide6](https://pypi.org/project/PySide6/).
The NER model is called from the *ner_extractor.py* script, where the entities to be extracted can be modified.
The LLM for text summarization is called from the *text_summarizer.py*. The local LLM in GGUF format is implemented with the Python binding for [llama.cpp](https://github.com/abetlen/llama-cpp-python). Before using the script, a model must be downloaded from [Huggingface](https://huggingface.co/) in GGUF format and put into the models folder.
After the *pdf_viewer.py* is started, there are multiple buttons visible in the GUI. First the PDF must be opened. Afterwards, both NER and text summarization can be applied separately. Here, the annotated PDF (NER) can be saved with all entities extracted and highlighted. Also, all extracted entities are displayed in a separate box on the right PDF viewer side. In the top right corner of the PDF Viewer, the text summary can be hidden and shown after being generated (only works after applying the *Summarize Text* button).
