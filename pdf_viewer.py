import os
import re
import sys
import fitz  # PyMuPDF
from llama_cpp import Llama
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QScrollArea, QLabel, QMessageBox, QTextEdit,
    QTreeWidget, QTreeWidgetItem
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush
from PySide6.QtCore import Qt, QThread, Signal
from ner_extractor import extract_entities
from text_summarizer import build_summarizer


# load the model (change file name if needed)
llm = Llama(
    model_path="models/qwen2.5-0.5b-instruct-q5_k_m.gguf",
    n_ctx=2048,           # context length
    n_threads=12           # adjust to your CPU
)

summarizer = build_summarizer(llm=llm)


# normalize text token for matching
def normalize_token(s: str) -> str:
    # remove surrounding punctuation and unicode whitespace, lower-case
    return re.sub(r'^[\W_]+|[\W_]+$', '', s.strip(), flags=re.UNICODE).lower()


def locate_entities_on_page(page, ner_out):
    """
    Given a PyMuPDF page and NER output (dict with 'entities'),
    return a list of {entity,label,box} for highlighting.
    """
    # extract words with positions: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    words = page.get_text("words")
    words = sorted(words, key=lambda w: (w[5], w[6], w[7]))

    # prepare token dicts
    tokens = []
    for idx, w in enumerate(words):
        tok = w[4]
        tokens.append({"text": tok, "norm": tok.strip().lower(), "idx": idx, "box": (w[0], w[1], w[2], w[3])})

    highlights = []

    ents = ner_out.get("entities", {}) if isinstance(ner_out, dict) else {}

    page_text = page.get_text("text")  # for fallback substring matching

    for label, ent_list in ents.items():
        for ent_text in ent_list:
            if not ent_text.strip():
                continue

            ent_words = [t.lower() for t in ent_text.split() if t.strip()]
            L = len(ent_words)
            matched = False

            # sliding window match on tokens
            for i in range(len(tokens) - L + 1):
                seq = [tokens[i + j]["norm"] for j in range(L)]
                if seq == ent_words:
                    x0s = [tokens[i + j]["box"][0] for j in range(L)]
                    y0s = [tokens[i + j]["box"][1] for j in range(L)]
                    x1s = [tokens[i + j]["box"][2] for j in range(L)]
                    y1s = [tokens[i + j]["box"][3] for j in range(L)]
                    bbox = (min(x0s), min(y0s), max(x1s), max(y1s))
                    highlights.append({"entity": ent_text, "label": label, "box": bbox})
                    matched = True

            # fallback: substring match in raw text
            if not matched:
                for m in re.finditer(re.escape(ent_text), page_text, flags=re.IGNORECASE):
                    char_index = m.start()
                    acc = ""
                    found_token_idx = None
                    for t in tokens:
                        if found_token_idx is None:
                            acc += (" " if acc else "") + t["text"]
                            if len(acc) >= char_index:
                                found_token_idx = t["idx"]
                                break
                    if found_token_idx is not None:
                        w = tokens[found_token_idx]
                        bbox = w["box"]
                        highlights.append({"entity": ent_text, "label": label, "box": bbox})

    return highlights



# NER across all pages (background thread)
class NERWorker(QThread):
    finished = Signal(object)  # can emit dict of highlights

    def __init__(self, doc_path: str, ner_function):
        super().__init__()
        self.doc_path = doc_path
        self.ner_function = ner_function

    def run(self):
        highlights_by_page = {}
        try:
            doc = fitz.open(self.doc_path)
            for pno, page in enumerate(doc):
                page_text = page.get_text("text")
                ner_out = self.ner_function(page_text)
                highlights_by_page[pno] = locate_entities_on_page(page, ner_out)
        except Exception as e:
            self.finished.emit({"__error__": str(e)})
            return

        self.finished.emit(highlights_by_page)


# summarize full PDF (background thread)
class SummaryWorker(QThread):
    finished = Signal(str)

    def __init__(self, doc_path: str, summarizer_function):
        super().__init__()
        self.doc_path = doc_path
        self.summarizer_function = summarizer_function

    def run(self):
        try:
            doc = fitz.open(self.doc_path)
            full_text = "\n".join([p.get_text("text") for p in doc])
            # call your summarizer function
            summary = self.summarizer_function(full_text)
            if isinstance(summary, list):
                # pipeline-style output
                summary_text = "\n".join([s.get("summary_text", str(s)) if isinstance(s, dict) else str(s) for s in summary])
            else:
                summary_text = str(summary)
        except Exception as e:
            summary_text = f"Summarizer error: {e}"

        self.finished.emit(summary_text)


# PDF Viewer widget (renders pages & overlays highlights)
class PDFViewer(QWidget):
    def __init__(self, zoom: float = 1.5):
        super().__init__()
        self.doc = None
        self.zoom = zoom
        self.page_pixmaps = []  # cached QPixmap
        self.highlights_by_page = {}  # page -> list of {entity,label,box}
        self.colors = {
            "person": QColor(255, 235, 59, 120),   # yellow-ish
            "company": QColor(100, 181, 246, 120), # blue-ish
            "product": QColor(129, 199, 132, 120), # green-ish
            "location": QColor(239, 83, 80, 120),  # red-ish
            "date": QColor(171, 71, 188, 120),     # purple-ish
            "default": QColor(255, 193, 7, 120)
        }

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(10)

    def load_pdf(self, path: str):
        self.doc_path = path
        self.doc = fitz.open(path)

        # clear previous
        for i in reversed(range(self.vlayout.count())):
            item = self.vlayout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()

        self.page_pixmaps = []
        self.page_texts = []            # <-- store extracted text
        self.highlights_by_page = {}

        for pno in range(self.doc.page_count):
            page = self.doc.load_page(pno)
            # extract text for NER
            text = page.get_text()
            self.page_texts.append(text)  # store for later
            # render page
            label = QLabel()
            label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            label.setContentsMargins(0, 0, 0, 0)

            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            qpix = QPixmap.fromImage(img)
            self.page_pixmaps.append((qpix, pix.width, pix.height, self.zoom))
            label.setPixmap(qpix)
            self.vlayout.addWidget(label)

        # add stretch
        self.vlayout.addStretch(1)


    def apply_highlights(self, highlights_by_page: dict, write_annotations: bool = False):
        """
        highlights_by_page: {pno: [ {entity,label,box}, ... ] }
        write_annotations: whether to add highlight annotations into the PDF file itself
        """
        if not self.doc:
            return

        self.highlights_by_page = highlights_by_page

        # optionally write annotations into the PDF (non-destructive in memory; must save to persist)
        if write_annotations:
            for pno, items in highlights_by_page.items():
                page = self.doc.load_page(pno)
                for it in items:
                    x0, y0, x1, y1 = it["box"]
                    rect = fitz.Rect(x0, y0, x1, y1)
                    try:
                        annot = page.add_highlight_annot(rect)
                        # set color lightly (annotation color uses 0..1 floats)
                        c = self.colors.get(it["label"], self.colors["default"])
                        annot.set_colors(stroke=(c.redF(), c.greenF(), c.blueF()))
                        annot.update()
                    except Exception:
                        # some PDFs require quads for highlight; skip silent
                        pass
            # save copy
            try:
                outpath = self.doc_path.replace(".pdf", "_annotated.pdf")
                self.doc.save(outpath, deflate=True)
            except Exception as e:
                print("Warning: could not save annotated PDF:", e)

        # re-render overlays on cached pixmaps
        # for each label widget in layout, overlay highlight boxes (scaled by zoom)
        # note: self.page_pixmaps entry order matches widgets order in layout
        widget_idx = 0
        for i in range(self.vlayout.count()):
            item = self.vlayout.itemAt(i)
            w = item.widget()
            if not isinstance(w, QLabel):
                continue
            if widget_idx >= len(self.page_pixmaps):
                break
            qpix, pw, ph, zoom = self.page_pixmaps[widget_idx]
            # copy base pixmap to draw overlays
            overlay_img = qpix.toImage().convertToFormat(QImage.Format_ARGB32)
            painter = QPainter(overlay_img)
            painter.setRenderHint(QPainter.Antialiasing)
            pno = widget_idx
            items = highlights_by_page.get(pno, [])
            for it in items:
                x0, y0, x1, y1 = it["box"]
                # scale boxes by zoom factor when drawing on pixmap
                sx0, sy0 = int(x0 * zoom), int(y0 * zoom)
                sx1, sy1 = int(x1 * zoom), int(y1 * zoom)
                w_rect = (sx0, sy0, sx1 - sx0, sy1 - sy0)
                color = self.colors.get(it["label"], self.colors["default"])
                painter.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0, color)
            painter.end()
            # set to label
            w.setPixmap(QPixmap.fromImage(overlay_img))
            widget_idx += 1


# main window with buttons
class MainWindow(QMainWindow):
    def __init__(self, ner_function, summarizer_function):
        super().__init__()
        self.setWindowTitle("PDF Viewer — NER & Summarizer")
        self.resize(1000, 800)

        self.ner_function = ner_function
        self.summarizer_function = summarizer_function
        self.current_path = None

        # central widget: scroll area containing viewer
        container = QWidget()
        container_layout = QVBoxLayout(container)
        toolbar = QHBoxLayout()
        container_layout.addLayout(toolbar)

        btn_open = QPushButton("Open PDF")
        btn_open.clicked.connect(self.open_pdf)
        toolbar.addWidget(btn_open)

        btn_ner = QPushButton("Highlight NER (all pages)")
        btn_ner.clicked.connect(self.do_ner)
        toolbar.addWidget(btn_ner)

        btn_save_ann = QPushButton("Highlight + Save PDF (annotated copy)")
        btn_save_ann.clicked.connect(self.do_ner_and_save)
        toolbar.addWidget(btn_save_ann)

        btn_sum = QPushButton("Summarize PDF")
        btn_sum.clicked.connect(self.do_summary)
        toolbar.addWidget(btn_sum)

        toolbar.addStretch(1)

        # viewer in scroll area
        self.viewer = PDFViewer(zoom=1.5)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.viewer)
        # viewer + entity Sidebar horizontal layout
        viewer_row = QHBoxLayout()
        # left side: the scroll area with PDF pages
        viewer_row.addWidget(self.scroll, stretch=4)
        # right side: entity sidebar
        self.entity_list = QTreeWidget()
        self.entity_list.setHeaderLabels(["Entities"])
        self.entity_list.setMinimumWidth(260)
        viewer_row.addWidget(self.entity_list, stretch=1)
        
        self.entity_list.itemClicked.connect(self.on_entity_clicked)
        container_layout.addLayout(viewer_row)

        # summary area (hidden until used)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.hide()
        container_layout.addWidget(self.summary_box)

        self.setCentralWidget(container)

        # threads references
        self.ner_worker = None
        self.summary_worker = None
    

    def on_entity_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        page, bbox = data
        
        # scroll the viewer to the page widget
        # each page widget is a QLabel added to self.viewer.vlayout in load_pdf()
        try:
            widget_index = page  # page is zero-based index matching layout order
            # vlayout contains widgets + final stretch; widget at index widget_index is the page label
            widget_item = self.viewer.vlayout.itemAt(widget_index)
            if widget_item is None:
                return
            widget = widget_item.widget()
            if widget is None:
                return
            # ensure the page widget is visible in the scroll area
            self.scroll.ensureWidgetVisible(widget)
        except Exception:
            # fallback - do nothing if scrolling fails
            pass

        # optionally flash the highlight if viewer supports it
        if hasattr(self.viewer, "flash_highlight"):
            try:
                self.viewer.flash_highlight(page, bbox)
            except Exception:
                pass

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF file", "", "PDF Files (*.pdf)")
        if not path:
            return
        self.current_path = path
        self.viewer.load_pdf(path)
        self.summary_box.hide()
        self.summary_box.clear()

    def do_ner(self):
        if not self.current_path:
            QMessageBox.warning(self, "No PDF", "Open a PDF first.")
            return
        self.setEnabled(False)
        self.ner_worker = NERWorker(self.current_path, self.ner_function)
        self.ner_worker.finished.connect(self.on_ner_finished)
        self.ner_worker.start()

    def do_ner_and_save(self):
        # same as do_ner but pass write_annotations True after results
        if not self.current_path:
            QMessageBox.warning(self, "No PDF", "Open a PDF first.")
            return
        self.setEnabled(False)
        self.ner_worker = NERWorker(self.current_path, self.ner_function)
        # use a lambda to capture results and then write
        self.ner_worker.finished.connect(lambda res: self.on_ner_finished(res, save=True))
        self.ner_worker.start()

    def on_ner_finished(self, result, save=False):
        self.setEnabled(True)
        if "__error__" in result:
            QMessageBox.critical(self, "NER error", str(result["__error__"]))
            return
        # result: {pno: [ {entity,label,box}, ... ], ... }
        self.viewer.apply_highlights(result, write_annotations=save)

        # populate sidebar
        self.entity_list.clear()  # reset previous

        # build a dict: {label: set of entities}
        entity_dict = {}
        for page_items in result.values():
            for item in page_items:
                label = item.get("label", "unknown")
                entity_text = item.get("entity", "")
                if label not in entity_dict:
                    entity_dict[label] = set()
                entity_dict[label].add(entity_text)
        
        # add top-level items per label
        for label, entities in entity_dict.items():
            top_item = QTreeWidgetItem([label])
            for entity in sorted(entities):
                child_item = QTreeWidgetItem([entity])
                # store the first page + bbox of that entity for navigation
                for page_no, page_items in result.items():
                    for page_item in page_items:
                        if page_item.get("entity") == entity:
                            child_item.setData(0, Qt.UserRole, (page_no, page_item.get("box")))
                            break
                    else:
                        continue
                    break
                top_item.addChild(child_item)
            self.entity_list.addTopLevelItem(top_item)
            top_item.setExpanded(False)  # collapsed by default

        total = sum(len(value) for value in result.values())
        QMessageBox.information(self, "NER finished", f"Found {total} entity occurrences across {len(result)} pages.")

    def do_summary(self):
        if not self.current_path:
            QMessageBox.warning(self, "No PDF", "Open a PDF first.")
            return
        self.setEnabled(False)
        self.summary_worker = SummaryWorker(self.current_path, self.summarizer_function)
        self.summary_worker.finished.connect(self.on_summary_finished)
        self.summary_worker.start()

    def on_summary_finished(self, summary_text):
        self.setEnabled(True)
        if summary_text is None:
            QMessageBox.critical(self, "Summary error", "No summary returned.")
            return
        # show in summary box
        self.summary_box.setPlainText(summary_text)
        self.summary_box.show()
        QMessageBox.information(self, "Summary finished", "Summary generated and shown below the viewer.")


# entry point
def main(ner_function, summarizer_function):
    app = QApplication(sys.argv)
    w = MainWindow(ner_function, summarizer_function)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # swap these for your real functions
    main(ner_function=extract_entities, summarizer_function=summarizer)