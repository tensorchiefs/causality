#!/bin/bash

# Set the working directory
cd /Users/oli/Documents/GitHub/causality

# Fifth run
R CMD BATCH --no-save --no-restore "--args 4 ls" summerof24/triangle_structured_continous.R output_4_ls.Rout
mv Rplots.pdf "triangle_plot_4_ls.pdf"

# Sixth run
R CMD BATCH --no-save --no-restore "--args 4 cs" summerof24/triangle_structured_continous.R output_4_cs.Rout
mv Rplots.pdf "triangle_plot_4_cs.pdf"
