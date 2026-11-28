"""Assemble the vector figures into a bookmarked PDF with page numbers."""
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

EXPERIMENT = Path(__file__).resolve().parents[1]
HERE = EXPERIMENT / "random" / "visuals"
OUTPUT = EXPERIMENT / "reports" / "DiffIon_One_vs_Three_Current_Figures.pdf"
ORDER = [
    ("main",0,"Method overview"),("extra",0,"A material’s measured inputs"),("main",1,"Current-band comparison"),
    ("main",2,"Learned estimators and gradient-free inversion"),("main",3,"Differentiable solver"),
    ("main",4,"PINNs from FNO + TTT"),("main",5,"PINNs from the center start"),("main",9,"Recovered transport curves"),
    ("main",6,"Current bands for learned and numerical methods"),("main",7,"Current bands for learned-start PINNs"),
    ("main",8,"Current bands for center-start PINNs"),("extra",1,"Compact optimizer benchmark"),
    ("extra",2,"Clean and noisy CNN results"),("extra",3,"Clean and noisy PINN results"),
]


def main():
    documents = {"main":PdfReader(HERE / "main_figures.pdf"), "extra":PdfReader(HERE / "secondary.pdf")}
    writer = PdfWriter()
    writer.add_metadata({"/Title":"DiffIon | One versus three current densities", "/Subject":"Paired transport-property benchmarks"})
    for index,(doc,number,name) in enumerate(ORDER):
        page = documents[doc].pages[number]
        width,height = map(float,(page.mediabox.width,page.mediabox.height))
        stream = BytesIO()
        overlay = Canvas(stream, pagesize=(width,height))
        overlay.setFont("Helvetica",8); overlay.setFillColor(HexColor("#60777B"))
        overlay.drawRightString(width-23,12,f"{index+1} / {len(ORDER)}")
        overlay.save()
        page.merge_page(PdfReader(stream).pages[0])
        writer.add_page(page)
        writer.add_outline_item(name,index)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("wb") as output:
        writer.write(output)
    assert len(PdfReader(OUTPUT).pages)==len(ORDER)
    print(OUTPUT)


if __name__=="__main__":
    main()
