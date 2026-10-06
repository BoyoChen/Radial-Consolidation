"""Portable vector overview without a TeX installation."""
import html
import io
import matplotlib.pyplot as plt
import pymupdf as fitz
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from . import paths
from .common import read_data, save_figure


def conceptual():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.5,2.4))
    ax.axis('off');ax.set(xlim=(-.15,3.7),ylim=(-.45,2.55))
    for x,y,color,lw in [(3,0,'#999999',1),(0,2,'#999999',1),(2.25,1.55,'#ae4439',2),
                          (2.25,.55,'#1a5d85',2),(2.25,0,'black',2)]:
        ax.annotate('',xy=(x,y),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=lw))
    ax.plot([2.25,2.25],[0,1.55],ls='--',lw=1,color='#999999')
    for x,y,text in [(3.1,0,r'$q_j$'),(-.08,2.17,'orthogonal'),(2.32,1.62,r'$u_j$'),
                      (2.4,.55,r'$\widetilde{u}_j(\alpha)$'),(1,-.3,r'$u_j^{\parallel}$: radial-only')]:
        ax.text(x,y,text,fontsize=9)
    save_figure(fig,'figure_01_method');plt.close(fig)
    fig=plt.figure(figsize=(5,2.2));ax=fig.add_axes([.02,.03,.96,.94]);ax.axis('off')
    lines=[('At formation boundary:',True),(r'$q\leftarrow\mathrm{normalize}(w)$',False),
        ('For each subsequent update:',True),(r'$u\leftarrow\mathrm{LocalSTDP}(\mathrm{spike\ traces})$',False),
        (r'$u\leftarrow\alpha u+(1-\alpha)q(q^{\mathsf{T}}u)$',False),
        (r'$w\leftarrow\mathrm{Regulate}(w+u)$',False)]
    for i,(text,bold) in enumerate(lines):
        ax.text(.01 if bold else .06,.94-i*.16,text,va='top',fontsize=11,fontweight='bold' if bold else 'normal')
    save_figure(fig,'algorithm_01_update');plt.close(fig)


def combine_direction():
    pdf=fitz.open()
    sources=[fitz.open(paths.OUT/f'kernel_direction_drift_{part}.pdf') for part in ['snapshots','angles']]
    width=sources[0][0].rect.width
    height=sum(s[0].rect.height for s in sources)+24
    page=pdf.new_page(width=width,height=height)
    y=12
    for letter,source in zip(['(a)','(b)'],sources):
        page.insert_text((2,y-3),letter,fontsize=8)
        h=source[0].rect.height
        page.show_pdf_page(fitz.Rect(0,y,width,y+h),source,0);y+=h+12
    pdf.save(paths.OUT/'figure_02_drift.pdf')
    pdf[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(paths.OUT/'figure_02_drift.png')
    for source in sources:source.close()
    pdf.close()


def build():
    conceptual();combine_direction()
    captions=read_data('captions.json')
    ordered=[('figure_01_method','Figure 1','fig:method'),('algorithm_01_update','Algorithm 1','alg:reference-update'),
        ('table_01_protocol','Table I','tab:protocol'),('figure_02_drift','Figure 2','fig:kernel-direction-drift'),
        ('table_02_geometry','Table II','tab:long'),('figure_03_kernel_homeostasis','Figure 3','fig:kernel-homeostasis'),
        ('table_03_results','Table III','tab:main'),('figure_04_accuracy_gap','Figure 4','fig:timing')]
    book=fitz.open()
    for stem,title,key in ordered:
        chart=fitz.open(paths.OUT/f'{stem}.pdf')
        width=720; chart_width=672
        chart_height=chart[0].rect.height*chart_width/chart[0].rect.width
        caption=Paragraph(html.escape(captions[key]),ParagraphStyle('caption',fontName='Times-Roman',fontSize=10,leading=13))
        _,cap_height=caption.wrap(chart_width,2000)
        height=chart_height+cap_height+85
        buffer=io.BytesIO();c=canvas.Canvas(buffer,pagesize=(width,height),invariant=1)
        c.setFont('Helvetica-Bold',13);c.drawString(24,height-24,title)
        caption.drawOn(c,24,20);c.showPage();c.save()
        page=fitz.open(stream=buffer.getvalue(),filetype='pdf')
        page[0].show_pdf_page(fitz.Rect(24,40,696,40+chart_height),chart,0)
        book.insert_pdf(page);chart.close();page.close()
    book.save(paths.OUT/'figures_and_tables.pdf',garbage=4,deflate=True)
    book.close()
