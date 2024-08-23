#!/bin/bash

# Set the working directory
cd /Users/oli/Documents/GitHub/causality

# First run
R CMD BATCH --no-save --no-restore "--args 1 ls" summerof24/triangle_structured_continous.R output_1_LS.Rout
mv Rplots.pdf "triangle_plot_1_ls.pdf"

# Second run
R CMD BATCH --no-save --no-restore "--args 1 cs" summerof24/triangle_structured_continous.R output_1_CS.Rout
mv Rplots.pdf "triangle_plot_1_cs.pdf"

# Third run
R CMD BATCH --no-save --no-restore "--args 2 ls" summerof24/triangle_structured_continous.R output_2_ls.Rout
mv Rplots.pdf "triangle_plot_2_ls.pdf"

# Fourth run
R CMD BATCH --no-save --no-restore "--args 2 cs" summerof24/triangle_structured_continous.R output_2_cs.Rout
mv Rplots.pdf "triangle_plot_2_cs.pdf"

# Fifth run
R CMD BATCH --no-save --no-restore "--args 3 ls" summerof24/triangle_structured_continous.R output_3_ls.Rout
mv Rplots.pdf "triangle_plot_3_ls.pdf"

# Sixth run
R CMD BATCH --no-save --no-restore "--args 3 cs" summerof24/triangle_structured_continous.R output_3_cs.Rout
mv Rplots.pdf "triangle_plot_3_cs.pdf"

# Fifth run
R CMD BATCH --no-save --no-restore "--args 4 ls" summerof24/triangle_structured_continous.R output_4_ls.Rout
mv Rplots.pdf "triangle_plot_4_ls.pdf"

# Sixth run
R CMD BATCH --no-save --no-restore "--args 4 cs" summerof24/triangle_structured_continous.R output_4_cs.Rout
mv Rplots.pdf "triangle_plot_4_cs.pdf"
