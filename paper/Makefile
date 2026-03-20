# Makefile for SozKZ paper
# Usage: make        (full build with bibliography)
#        make quick  (single pdflatex pass)
#        make clean  (remove build artifacts)
#        make assets (regenerate Phase 2 figures/tables/macros)

MAIN = main

.PHONY: all quick clean assets

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex abstract.tex introduction.tex related.tex methodology.tex experiments.tex results.tex conclusion.tex references.bib macros.tex
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

quick:
	pdflatex -interaction=nonstopmode $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf *.fls *.fdb_latexmk *.synctex.gz

assets:
	cd .. && PYTHONPATH=. python scripts/analysis/generate_all.py
