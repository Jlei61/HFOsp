"""Keep Noto OTF readable in PDFs even when legacy plotters reset rcParams."""
def install():
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    original=Figure.savefig
    if not getattr(original,'_topic4_type3_pdf',False):
        def savefig(self,fname,*args,**kwargs):
            is_pdf=str(fname).lower().endswith('.pdf') or kwargs.get('format')=='pdf'
            if is_pdf:
                with mpl.rc_context({'pdf.fonttype':3}):
                    return original(self,fname,*args,**kwargs)
            return original(self,fname,*args,**kwargs)
        savefig._topic4_type3_pdf=True
        Figure.savefig=savefig
    # PdfPages embeds fonts only on close, after individual savefig contexts end.
    close=PdfPages.close
    if not getattr(close,'_topic4_type3_pdf',False):
        def close_type3(self):
            with mpl.rc_context({'pdf.fonttype':3}):return close(self)
        close_type3._topic4_type3_pdf=True
        PdfPages.close=close_type3
